# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for the WALNUTS kernel.

WALNUTS extends NUTS with a fixed macro time grid and a locally refined
dyadic leapfrog step size inside each macro interval.  The implementation below
keeps the same public shape as :mod:`blackjax.mcmc.nuts`, but uses a streaming
bottom-up subtree representation so orbit construction remains JIT-compatible
and O(max tree depth) in memory.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.types import ArrayTree, PRNGKey

__all__ = [
    "WALNUTSInfo",
    "WALNUTSDiagnostics",
    "MicroStepInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
    "iterative_walnuts_proposal",
    "micro_step",
    "micro_step_pmf",
]


init = hmc.init


class WALNUTSInfo(NamedTuple):
    """Additional information on a WALNUTS transition.

    momentum:
        Momentum sampled at the start of the transition.
    is_divergent:
        Whether any generated macro step exceeded ``divergence_threshold``.
    is_turning:
        Whether orbit construction stopped because of a U-turn or sub-U-turn.
    energy:
        Hamiltonian energy of the selected proposal.
    trajectory_leftmost_state:
        Leftmost state in the accepted accumulated orbit.
    trajectory_rightmost_state:
        Rightmost state in the accepted accumulated orbit.
    num_trajectory_expansions:
        Number of doubling expansions attempted.
    num_macro_steps:
        Number of macro-grid states generated, including the initial state.
    num_integration_steps:
        Number of leapfrog micro-steps used for micro searches, final macro
        integrations, and reverse-probability checks.
    acceptance_rate:
        Average Metropolis-style energy diagnostic over generated leaves.
    max_micro_steps:
        Largest number of leapfrog micro-steps selected inside one macro step.
    min_micro_step_size:
        Smallest selected leapfrog step size inside the transition.
    max_micro_step_size:
        Largest selected leapfrog step size inside the transition.
    no_refinement_rate:
        Fraction of generated macro steps whose critical micro level was 0.
    hit_max_micro_doublings:
        Whether a micro search reached ``max_num_micro_doublings``.
    """

    momentum: ArrayTree
    is_divergent: bool
    is_turning: bool
    energy: float
    trajectory_leftmost_state: integrators.IntegratorState
    trajectory_rightmost_state: integrators.IntegratorState
    num_trajectory_expansions: int
    num_macro_steps: int
    num_integration_steps: int
    acceptance_rate: float
    max_micro_steps: int
    min_micro_step_size: float
    max_micro_step_size: float
    no_refinement_rate: float
    hit_max_micro_doublings: bool


class MicroStepInfo(NamedTuple):
    """Information returned by the bounded dyadic micro-step search."""

    level: int
    num_integration_steps: int
    hit_max_micro_doublings: bool


class WALNUTSDiagnostics(NamedTuple):
    """Aggregated diagnostics collected while building a WALNUTS orbit."""

    num_macro_steps: int
    num_integration_steps: int
    max_micro_steps: int
    min_micro_steps: int
    num_no_refinement: int
    hit_max_micro_doublings: bool
    sum_log_p_accept: float


class _Subtree(NamedTuple):
    left_state: integrators.IntegratorState
    right_state: integrators.IntegratorState
    selected_state: integrators.IntegratorState
    log_weight: float


class _SubtreeStack(NamedTuple):
    left_state: integrators.IntegratorState
    right_state: integrators.IntegratorState
    selected_state: integrators.IntegratorState
    log_weight: jax.Array
    is_valid: jax.Array


class _SubtreeResult(NamedTuple):
    subtree: _Subtree
    log_weight_end: float
    is_turning: bool
    is_divergent: bool
    diagnostics: WALNUTSDiagnostics


def _as_int32(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _micro_steps_from_level(level):
    return jnp.left_shift(_as_int32(1), _as_int32(level))


def _energy(
    kinetic_energy: metrics.KineticEnergy, state: integrators.IntegratorState
):
    return -state.logdensity + kinetic_energy(state.momentum, position=state.position)


def _flip_momentum(state: integrators.IntegratorState) -> integrators.IntegratorState:
    momentum = jax.tree.map(lambda m: -m, state.momentum)
    return integrators.IntegratorState(
        state.position, momentum, state.logdensity, state.logdensity_grad
    )


def _integrate_fixed_level(
    integrator: Callable,
    state: integrators.IntegratorState,
    step_size,
    level,
) -> integrators.IntegratorState:
    num_steps = _micro_steps_from_level(level)
    micro_step_size = step_size / num_steps

    def do_step(i, current_state):
        del i
        return integrator(current_state, micro_step_size)

    return jax.lax.fori_loop(0, num_steps, do_step, state)


def _integrate_until_invalid(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    state: integrators.IntegratorState,
    step_size,
    level,
    energy_threshold,
):
    """Try one dyadic refinement level, stopping early if its swing is too large."""

    num_steps = _micro_steps_from_level(level)
    micro_step_size = step_size / num_steps
    initial_energy = _energy(kinetic_energy, state)

    def keep_going(carry):
        i, _, energy_min, energy_max, is_invalid = carry
        energy_swing = energy_max - energy_min
        return (i < num_steps) & ~is_invalid & (energy_swing <= energy_threshold)

    def step_once(carry):
        i, current_state, energy_min, energy_max, _ = carry
        new_state = integrator(current_state, micro_step_size)
        new_energy = _energy(kinetic_energy, new_state)
        energy_min = jnp.minimum(energy_min, new_energy)
        energy_max = jnp.maximum(energy_max, new_energy)
        is_invalid = (energy_max - energy_min) > energy_threshold
        is_invalid = is_invalid | ~jnp.isfinite(new_energy)
        return i + 1, new_state, energy_min, energy_max, is_invalid

    initial_carry = (
        _as_int32(0),
        state,
        initial_energy,
        initial_energy,
        ~jnp.isfinite(initial_energy),
    )
    i, candidate_state, energy_min, energy_max, is_invalid = jax.lax.while_loop(
        keep_going, step_once, initial_carry
    )
    energy_swing = energy_max - energy_min
    is_invalid = is_invalid | (energy_swing > energy_threshold)
    return candidate_state, energy_swing, i, is_invalid


def micro_step(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    state: integrators.IntegratorState,
    step_size,
    energy_threshold,
    max_num_micro_doublings: int = 8,
) -> MicroStepInfo:
    """Find the smallest dyadic refinement level satisfying the energy bound.

    The search is bounded by ``max_num_micro_doublings`` to keep compilation
    shapes static and to avoid unbounded work in pathological regions.
    """

    max_level = _as_int32(max_num_micro_doublings)

    def keep_searching(carry):
        level, has_valid_level, *_ = carry
        return (level <= max_level) & ~has_valid_level

    def try_level(carry):
        level, has_valid_level, best_level, total_steps, hit_max = carry
        _, _, steps_taken, is_invalid = _integrate_until_invalid(
            integrator,
            kinetic_energy,
            state,
            step_size,
            level,
            energy_threshold,
        )
        is_valid = ~is_invalid
        best_level = jnp.where(is_valid, level, best_level)
        has_valid_level = has_valid_level | is_valid
        hit_max = hit_max | (level == max_level)
        return (
            level + 1,
            has_valid_level,
            best_level,
            total_steps + steps_taken,
            hit_max,
        )

    initial_carry = (
        _as_int32(0),
        jnp.asarray(False),
        max_level,
        _as_int32(0),
        jnp.asarray(False),
    )
    _, _, level, num_steps, hit_max = jax.lax.while_loop(
        keep_searching, try_level, initial_carry
    )
    return MicroStepInfo(level, num_steps, hit_max)


def micro_step_pmf(
    selected_level,
    base_level,
    max_num_micro_doublings: int = 8,
    micro_step_distribution: str = "r2p",
):
    """Probability of selecting ``selected_level`` from a critical level."""

    selected_level = _as_int32(selected_level)
    base_level = _as_int32(base_level)
    max_level = _as_int32(max_num_micro_doublings)

    if micro_step_distribution == "deterministic":
        return jnp.where(selected_level == base_level, 1.0, 0.0)
    if micro_step_distribution != "r2p":
        raise ValueError(
            "micro_step_distribution must be either 'r2p' or 'deterministic'"
        )

    at_cap = base_level >= max_level
    return jnp.where(
        at_cap,
        jnp.where(selected_level == max_level, 1.0, 0.0),
        jnp.where(
            selected_level == base_level,
            2.0 / 3.0,
            jnp.where(selected_level == base_level + 1, 1.0 / 3.0, 0.0),
        ),
    )


def _sample_micro_level(
    rng_key: PRNGKey,
    base_level,
    max_num_micro_doublings: int,
    micro_step_distribution: str,
):
    if micro_step_distribution == "deterministic":
        return base_level
    if micro_step_distribution != "r2p":
        raise ValueError(
            "micro_step_distribution must be either 'r2p' or 'deterministic'"
        )

    choose_finer = jax.random.bernoulli(rng_key, 1.0 / 3.0)
    proposed_level = base_level + choose_finer.astype(jnp.int32)
    return jnp.minimum(proposed_level, _as_int32(max_num_micro_doublings))


def _new_diagnostics(max_num_micro_doublings: int) -> WALNUTSDiagnostics:
    max_possible_micro_steps = 1 << max_num_micro_doublings
    return WALNUTSDiagnostics(
        _as_int32(0),
        _as_int32(0),
        _as_int32(0),
        _as_int32(max_possible_micro_steps),
        _as_int32(0),
        jnp.asarray(False),
        -jnp.inf,
    )


def _add_diagnostics(
    left: WALNUTSDiagnostics, right: WALNUTSDiagnostics
) -> WALNUTSDiagnostics:
    return WALNUTSDiagnostics(
        left.num_macro_steps + right.num_macro_steps,
        left.num_integration_steps + right.num_integration_steps,
        jnp.maximum(left.max_micro_steps, right.max_micro_steps),
        jnp.minimum(left.min_micro_steps, right.min_micro_steps),
        left.num_no_refinement + right.num_no_refinement,
        left.hit_max_micro_doublings | right.hit_max_micro_doublings,
        jnp.logaddexp(left.sum_log_p_accept, right.sum_log_p_accept),
    )


def _state_where(
    predicate,
    true_state: integrators.IntegratorState,
    false_state: integrators.IntegratorState,
) -> integrators.IntegratorState:
    return jax.tree.map(lambda x, y: jnp.where(predicate, x, y), true_state, false_state)


def _new_state_stack(
    state: integrators.IntegratorState, num_levels: int
) -> integrators.IntegratorState:
    def zeros_like_with_level(x):
        x = jnp.asarray(x)
        return jnp.zeros((num_levels,) + jnp.shape(x), dtype=x.dtype)

    return integrators.IntegratorState(
        jax.tree.map(zeros_like_with_level, state.position),
        jax.tree.map(zeros_like_with_level, state.momentum),
        zeros_like_with_level(state.logdensity),
        jax.tree.map(zeros_like_with_level, state.logdensity_grad),
    )


def _get_stacked_state(stack_state: integrators.IntegratorState, level):
    return jax.tree.map(lambda x: x[level], stack_state)


def _set_stacked_state(
    stack_state: integrators.IntegratorState,
    level,
    state: integrators.IntegratorState,
) -> integrators.IntegratorState:
    return jax.tree.map(lambda xs, x: xs.at[level].set(x), stack_state, state)


def _new_subtree_stack(
    state: integrators.IntegratorState, num_levels: int
) -> _SubtreeStack:
    state_stack = _new_state_stack(state, num_levels)
    return _SubtreeStack(
        state_stack,
        state_stack,
        state_stack,
        jnp.full((num_levels,), -jnp.inf),
        jnp.zeros((num_levels,), dtype=bool),
    )


def _get_subtree(stack: _SubtreeStack, level) -> _Subtree:
    return _Subtree(
        _get_stacked_state(stack.left_state, level),
        _get_stacked_state(stack.right_state, level),
        _get_stacked_state(stack.selected_state, level),
        stack.log_weight[level],
    )


def _set_subtree(stack: _SubtreeStack, level, subtree: _Subtree) -> _SubtreeStack:
    return _SubtreeStack(
        _set_stacked_state(stack.left_state, level, subtree.left_state),
        _set_stacked_state(stack.right_state, level, subtree.right_state),
        _set_stacked_state(stack.selected_state, level, subtree.selected_state),
        stack.log_weight.at[level].set(subtree.log_weight),
        stack.is_valid.at[level].set(True),
    )


def _invalidate_subtree(stack: _SubtreeStack, level) -> _SubtreeStack:
    return _SubtreeStack(
        stack.left_state,
        stack.right_state,
        stack.selected_state,
        stack.log_weight,
        stack.is_valid.at[level].set(False),
    )


def _metric_dot(
    metric: metrics.Metric,
    position,
    left,
    right,
):
    scaled_left = metric.scale(position, left, inv=True, trans=True)
    scaled_right = metric.scale(position, right, inv=True, trans=True)
    flat_left, _ = ravel_pytree(scaled_left)
    flat_right, _ = ravel_pytree(scaled_right)
    return jnp.dot(flat_left, flat_right)


def _is_uturn(
    metric: metrics.Metric,
    left_state: integrators.IntegratorState,
    right_state: integrators.IntegratorState,
):
    delta_position = jax.tree.map(
        jnp.subtract, right_state.position, left_state.position
    )
    turning_at_left = (
        _metric_dot(metric, left_state.position, left_state.momentum, delta_position)
        < 0
    )
    turning_at_right = (
        _metric_dot(metric, right_state.position, right_state.momentum, delta_position)
        < 0
    )
    return turning_at_left | turning_at_right


def _barker_pick_right(rng_key, log_weight_left, log_weight_right):
    log_total = jnp.logaddexp(log_weight_left, log_weight_right)
    probability_right = jnp.where(
        jnp.isneginf(log_total), 0.0, jnp.exp(log_weight_right - log_total)
    )
    return jax.random.bernoulli(rng_key, probability_right)


def _merge_subtrees(
    rng_key: PRNGKey,
    metric: metrics.Metric,
    direction,
    left_in_iteration: _Subtree,
    right_in_iteration: _Subtree,
) -> tuple[_Subtree, bool]:
    left_in_time, right_in_time = jax.lax.cond(
        direction > 0,
        lambda: (left_in_iteration, right_in_iteration),
        lambda: (right_in_iteration, left_in_iteration),
    )
    is_turning = _is_uturn(metric, left_in_time.left_state, right_in_time.right_state)

    pick_right = _barker_pick_right(
        rng_key, left_in_iteration.log_weight, right_in_iteration.log_weight
    )
    selected_state = _state_where(
        pick_right,
        right_in_iteration.selected_state,
        left_in_iteration.selected_state,
    )
    merged_log_weight = jnp.logaddexp(
        left_in_iteration.log_weight, right_in_iteration.log_weight
    )
    return (
        _Subtree(
            left_in_time.left_state,
            right_in_time.right_state,
            selected_state,
            merged_log_weight,
        ),
        is_turning,
    )


def _build_leaf(
    rng_key: PRNGKey,
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    state: integrators.IntegratorState,
    log_weight_start,
    initial_log_weight,
    direction,
    step_size,
    energy_threshold,
    max_num_micro_doublings: int,
    micro_step_distribution: str,
    divergence_threshold,
) -> tuple[integrators.IntegratorState, float, bool, WALNUTSDiagnostics]:
    key_level, _ = jax.random.split(rng_key)
    old_energy = _energy(kinetic_energy, state)

    def forward_leaf():
        forward_micro = micro_step(
            integrator,
            kinetic_energy,
            state,
            step_size,
            energy_threshold,
            max_num_micro_doublings,
        )
        selected_level = _sample_micro_level(
            key_level,
            forward_micro.level,
            max_num_micro_doublings,
            micro_step_distribution,
        )
        new_state = _integrate_fixed_level(integrator, state, step_size, selected_level)
        reverse_state = _flip_momentum(new_state)
        reverse_micro = micro_step(
            integrator,
            kinetic_energy,
            reverse_state,
            step_size,
            energy_threshold,
            max_num_micro_doublings,
        )
        return new_state, selected_level, forward_micro, reverse_micro

    def backward_leaf():
        flipped_state = _flip_momentum(state)
        backward_micro = micro_step(
            integrator,
            kinetic_energy,
            flipped_state,
            step_size,
            energy_threshold,
            max_num_micro_doublings,
        )
        selected_level = _sample_micro_level(
            key_level,
            backward_micro.level,
            max_num_micro_doublings,
            micro_step_distribution,
        )
        integrated_state = _integrate_fixed_level(
            integrator, flipped_state, step_size, selected_level
        )
        new_state = _flip_momentum(integrated_state)
        reverse_micro = micro_step(
            integrator,
            kinetic_energy,
            new_state,
            step_size,
            energy_threshold,
            max_num_micro_doublings,
        )
        return new_state, selected_level, backward_micro, reverse_micro

    new_state, selected_level, forward_micro, reverse_micro = jax.lax.cond(
        direction > 0, forward_leaf, backward_leaf
    )

    p_forward = micro_step_pmf(
        selected_level,
        forward_micro.level,
        max_num_micro_doublings,
        micro_step_distribution,
    )
    p_reverse = micro_step_pmf(
        selected_level,
        reverse_micro.level,
        max_num_micro_doublings,
        micro_step_distribution,
    )

    new_energy = _energy(kinetic_energy, new_state)
    is_divergent = (new_energy - old_energy) > divergence_threshold
    is_divergent = is_divergent | ~jnp.isfinite(new_energy)

    has_valid_reverse = (p_reverse > 0.0) & (p_forward > 0.0)
    log_weight = (
        log_weight_start
        + old_energy
        - new_energy
        + jnp.log(p_reverse)
        - jnp.log(p_forward)
    )
    log_weight = jnp.where(
        has_valid_reverse & ~is_divergent & jnp.isfinite(log_weight),
        log_weight,
        -jnp.inf,
    )

    selected_micro_steps = _micro_steps_from_level(selected_level)
    no_refinement = forward_micro.level == 0
    hit_max = (
        forward_micro.hit_max_micro_doublings
        | reverse_micro.hit_max_micro_doublings
        | (selected_level == max_num_micro_doublings)
    )
    num_integration_steps = (
        forward_micro.num_integration_steps
        + selected_micro_steps
        + reverse_micro.num_integration_steps
    )
    log_p_accept = jnp.minimum(log_weight - initial_log_weight, 0.0)
    diagnostics = WALNUTSDiagnostics(
        _as_int32(1),
        num_integration_steps,
        selected_micro_steps,
        selected_micro_steps,
        no_refinement.astype(jnp.int32),
        hit_max,
        log_p_accept,
    )

    return new_state, log_weight, is_divergent, diagnostics


def _build_subtree(
    rng_key: PRNGKey,
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    metric: metrics.Metric,
    initial_state: integrators.IntegratorState,
    log_weight_start,
    initial_log_weight,
    direction,
    depth,
    step_size,
    energy_threshold,
    max_num_doublings: int,
    max_num_micro_doublings: int,
    micro_step_distribution: str,
    divergence_threshold,
) -> _SubtreeResult:
    num_levels = max_num_doublings + 1
    max_num_steps = _micro_steps_from_level(depth)
    stack = _new_subtree_stack(initial_state, num_levels)
    initial_subtree = _Subtree(initial_state, initial_state, initial_state, -jnp.inf)
    diagnostics = _new_diagnostics(max_num_micro_doublings)

    def keep_building(carry):
        k, *_rest, done = carry
        return (k < max_num_steps) & ~done

    def build_one_leaf(carry):
        (
            k,
            current_state,
            current_log_weight,
            stack,
            root_subtree,
            root_log_weight_end,
            is_turning,
            is_divergent,
            diagnostics,
            _,
        ) = carry

        leaf_key = jax.random.fold_in(rng_key, k)
        (
            new_state,
            new_log_weight,
            leaf_is_divergent,
            leaf_diagnostics,
        ) = _build_leaf(
            leaf_key,
            integrator,
            kinetic_energy,
            current_state,
            current_log_weight,
            initial_log_weight,
            direction,
            step_size,
            energy_threshold,
            max_num_micro_doublings,
            micro_step_distribution,
            divergence_threshold,
        )
        diagnostics = _add_diagnostics(diagnostics, leaf_diagnostics)
        current_subtree = _Subtree(new_state, new_state, new_state, new_log_weight)

        def keep_merging(merge_carry):
            level, _, stack, subtree, turning = merge_carry
            should_merge = (level < depth) & stack.is_valid[level] & ~turning
            return should_merge

        def merge_once(merge_carry):
            level, merge_key, stack, subtree, turning = merge_carry
            pending_subtree = _get_subtree(stack, level)
            merged_subtree, turning_here = _merge_subtrees(
                jax.random.fold_in(merge_key, level),
                metric,
                direction,
                pending_subtree,
                subtree,
            )
            stack = _invalidate_subtree(stack, level)
            return (
                level + 1,
                merge_key,
                stack,
                merged_subtree,
                turning | turning_here,
            )

        merge_key = jax.random.fold_in(rng_key, max_num_steps + k)
        (
            subtree_level,
            _,
            stack,
            current_subtree,
            subtree_is_turning,
        ) = jax.lax.while_loop(
            keep_merging,
            merge_once,
            (_as_int32(0), merge_key, stack, current_subtree, jnp.asarray(False)),
        )

        done = subtree_is_turning | leaf_is_divergent

        def finish_now():
            return stack, current_subtree

        def park_subtree():
            return _set_subtree(stack, subtree_level, current_subtree), root_subtree

        stack, root_subtree = jax.lax.cond(done, finish_now, park_subtree)
        root_log_weight_end = jnp.where(done, new_log_weight, root_log_weight_end)
        is_turning = is_turning | subtree_is_turning
        is_divergent = is_divergent | leaf_is_divergent

        return (
            k + 1,
            new_state,
            new_log_weight,
            stack,
            root_subtree,
            root_log_weight_end,
            is_turning,
            is_divergent,
            diagnostics,
            done,
        )

    initial_carry = (
        _as_int32(0),
        initial_state,
        log_weight_start,
        stack,
        initial_subtree,
        log_weight_start,
        jnp.asarray(False),
        jnp.asarray(False),
        diagnostics,
        jnp.asarray(False),
    )
    (
        _,
        _,
        final_log_weight,
        stack,
        root_subtree,
        root_log_weight_end,
        is_turning,
        is_divergent,
        diagnostics,
        stopped_early,
    ) = jax.lax.while_loop(keep_building, build_one_leaf, initial_carry)

    final_subtree = jax.lax.cond(
        stopped_early,
        lambda: root_subtree,
        lambda: _get_subtree(stack, depth),
    )
    root_log_weight_end = jnp.where(stopped_early, root_log_weight_end, final_log_weight)

    return _SubtreeResult(
        final_subtree,
        root_log_weight_end,
        is_turning,
        is_divergent,
        diagnostics,
    )


def iterative_walnuts_proposal(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    metric: metrics.Metric,
    max_num_doublings: int = 10,
    max_num_micro_doublings: int = 8,
    energy_threshold: float = 0.3,
    micro_step_distribution: str = "r2p",
    divergence_threshold: float = 1000,
) -> Callable:
    """Build the iterative WALNUTS proposal generator."""

    if micro_step_distribution not in {"r2p", "deterministic"}:
        raise ValueError(
            "micro_step_distribution must be either 'r2p' or 'deterministic'"
        )

    def propose(rng_key, initial_state: integrators.IntegratorState, step_size):
        initial_energy = _energy(kinetic_energy, initial_state)
        initial_log_weight = -initial_energy
        initial_subtree = _Subtree(
            initial_state, initial_state, initial_state, initial_log_weight
        )
        diagnostics = _new_diagnostics(max_num_micro_doublings)

        def keep_expanding(carry):
            step, *_rest, is_turning, is_divergent, _diagnostics = carry
            return (step < max_num_doublings) & ~is_turning & ~is_divergent

        def expand_once(carry):
            (
                step,
                global_subtree,
                selected_state,
                global_log_weight,
                forward_log_weight_end,
                backward_log_weight_end,
                is_turning,
                is_divergent,
                diagnostics,
            ) = carry

            subkey = jax.random.fold_in(rng_key, step)
            direction_key, subtree_key, proposal_key = jax.random.split(subkey, 3)
            direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)

            start_state = jax.lax.cond(
                direction > 0,
                lambda: global_subtree.right_state,
                lambda: global_subtree.left_state,
            )
            log_weight_start = jnp.where(
                direction > 0, forward_log_weight_end, backward_log_weight_end
            )

            subtree_result = _build_subtree(
                subtree_key,
                integrator,
                kinetic_energy,
                metric,
                start_state,
                log_weight_start,
                initial_log_weight,
                direction,
                step,
                step_size,
                energy_threshold,
                max_num_doublings,
                max_num_micro_doublings,
                micro_step_distribution,
                divergence_threshold,
            )
            diagnostics = _add_diagnostics(
                diagnostics, subtree_result.diagnostics
            )

            should_stop_before_merge = (
                subtree_result.is_turning | subtree_result.is_divergent
            )

            def stop_without_merge():
                return (
                    global_subtree,
                    selected_state,
                    global_log_weight,
                    forward_log_weight_end,
                    backward_log_weight_end,
                    subtree_result.is_turning,
                    subtree_result.is_divergent,
                )

            def merge_extension():
                log_accept = subtree_result.subtree.log_weight - global_log_weight
                do_accept = jnp.log(jax.random.uniform(proposal_key)) <= log_accept
                new_selected_state = _state_where(
                    do_accept, subtree_result.subtree.selected_state, selected_state
                )
                merged_subtree, global_is_turning = _merge_subtrees(
                    proposal_key,
                    metric,
                    direction,
                    global_subtree,
                    subtree_result.subtree,
                )
                new_global_log_weight = jnp.logaddexp(
                    global_log_weight, subtree_result.subtree.log_weight
                )
                new_forward_log_weight_end = jnp.where(
                    direction > 0,
                    subtree_result.log_weight_end,
                    forward_log_weight_end,
                )
                new_backward_log_weight_end = jnp.where(
                    direction > 0,
                    backward_log_weight_end,
                    subtree_result.log_weight_end,
                )
                return (
                    merged_subtree,
                    new_selected_state,
                    new_global_log_weight,
                    new_forward_log_weight_end,
                    new_backward_log_weight_end,
                    global_is_turning,
                    jnp.asarray(False),
                )

            (
                global_subtree,
                selected_state,
                global_log_weight,
                forward_log_weight_end,
                backward_log_weight_end,
                is_turning_update,
                is_divergent_update,
            ) = jax.lax.cond(
                should_stop_before_merge, stop_without_merge, merge_extension
            )

            return (
                step + 1,
                global_subtree,
                selected_state,
                global_log_weight,
                forward_log_weight_end,
                backward_log_weight_end,
                is_turning | is_turning_update,
                is_divergent | is_divergent_update,
                diagnostics,
            )

        initial_carry = (
            _as_int32(0),
            initial_subtree,
            initial_state,
            initial_log_weight,
            initial_log_weight,
            initial_log_weight,
            jnp.asarray(False),
            jnp.asarray(False),
            diagnostics,
        )
        (
            num_expansions,
            global_subtree,
            selected_state,
            _,
            _,
            _,
            is_turning,
            is_divergent,
            diagnostics,
        ) = jax.lax.while_loop(keep_expanding, expand_once, initial_carry)

        generated_macro_steps = jnp.maximum(diagnostics.num_macro_steps, 1)
        num_macro_steps = diagnostics.num_macro_steps + 1
        no_refinement_rate = (
            diagnostics.num_no_refinement / generated_macro_steps.astype(jnp.float32)
        )
        acceptance_rate = (
            jnp.exp(diagnostics.sum_log_p_accept)
            / generated_macro_steps.astype(jnp.float32)
        )

        max_micro_steps = jnp.maximum(diagnostics.max_micro_steps, 1)
        min_micro_steps = jnp.maximum(diagnostics.min_micro_steps, 1)
        min_micro_step_size = step_size / max_micro_steps
        max_micro_step_size = step_size / min_micro_steps
        proposal_energy = _energy(kinetic_energy, selected_state)

        info = WALNUTSInfo(
            initial_state.momentum,
            is_divergent,
            is_turning,
            proposal_energy,
            global_subtree.left_state,
            global_subtree.right_state,
            num_expansions,
            num_macro_steps,
            diagnostics.num_integration_steps,
            acceptance_rate,
            max_micro_steps,
            min_micro_step_size,
            max_micro_step_size,
            no_refinement_rate,
            diagnostics.hit_max_micro_doublings,
        )
        return selected_state, info

    return propose


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
):
    """Build a WALNUTS kernel."""

    def kernel(
        rng_key: PRNGKey,
        state: hmc.HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        energy_threshold: float = 0.3,
        max_num_doublings: int = 10,
        max_num_micro_doublings: int = 8,
        micro_step_distribution: str = "r2p",
    ) -> tuple[hmc.HMCState, WALNUTSInfo]:
        """Generate a new sample with the WALNUTS kernel."""

        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator = iterative_walnuts_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            metric,
            max_num_doublings,
            max_num_micro_doublings,
            energy_threshold,
            micro_step_distribution,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)
        position, logdensity, logdensity_grad = state
        momentum = metric.sample_momentum(key_momentum, position)
        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state, step_size)
        proposal = hmc.HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )
        return proposal, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    *,
    energy_threshold: float = 0.3,
    max_num_doublings: int = 10,
    max_num_micro_doublings: int = 8,
    micro_step_distribution: str = "r2p",
    divergence_threshold: float = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """Implements the user interface for the WALNUTS kernel."""

    kernel = build_kernel(integrator, divergence_threshold)
    metric = metrics.default_metric(inverse_mass_matrix)
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(
            step_size,
            metric,
            energy_threshold,
            max_num_doublings,
            max_num_micro_doublings,
            micro_step_distribution,
        ),
    )
