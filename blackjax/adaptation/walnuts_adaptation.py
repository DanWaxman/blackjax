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
"""Warmup for the WALNUTS macro step size."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

__all__ = ["WALNUTSAdaptationState", "base", "walnuts_adaptation"]


class WALNUTSAdaptationState(NamedTuple):
    """Adaptation state for WALNUTS macro step-size warmup."""

    ss_state: DualAveragingAdaptationState
    step_size: float
    inverse_mass_matrix: Array


def base(
    target_no_refinement_rate: float = 0.8,
) -> tuple[Callable, Callable, Callable]:
    """Tune the WALNUTS macro step size.

    The adaptation statistic is ``info.no_refinement_rate``: the fraction of
    macro intervals whose critical micro refinement level was zero.  Treating
    that statistic like an acceptance rate gives the desired behavior under
    dual averaging: if no-refinement is too common, increase ``step_size``;
    if refinement is too common, decrease it.
    """

    da_init, da_update, da_final = dual_averaging_adaptation(
        target_no_refinement_rate
    )

    def init(
        position: ArrayLikeTree,
        initial_step_size: float,
        inverse_mass_matrix: Array | None = None,
    ) -> WALNUTSAdaptationState:
        if inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(pytree_size(position))
        return WALNUTSAdaptationState(
            da_init(initial_step_size),
            initial_step_size,
            inverse_mass_matrix,
        )

    def update(
        adaptation_state: WALNUTSAdaptationState,
        no_refinement_rate: float,
    ) -> WALNUTSAdaptationState:
        new_ss_state = da_update(adaptation_state.ss_state, no_refinement_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)
        return WALNUTSAdaptationState(
            new_ss_state,
            new_step_size,
            adaptation_state.inverse_mass_matrix,
        )

    def final(adaptation_state: WALNUTSAdaptationState):
        return da_final(adaptation_state.ss_state), adaptation_state.inverse_mass_matrix

    return init, update, final


def walnuts_adaptation(
    logdensity_fn: Callable,
    inverse_mass_matrix: Array | None = None,
    initial_step_size: float = 1.0,
    target_no_refinement_rate: float = 0.8,
    progress_bar: bool = False,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the macro step size for WALNUTS.

    This warmup keeps ``energy_threshold`` fixed and only adapts the macro step
    size ``h``.  It is intentionally lighter than full window adaptation and
    mirrors Appendix C's practical objective of targeting a chosen probability
    that ``micro`` accepts the macro step without refinement.
    """

    mcmc_kernel = mcmc.walnuts.build_kernel(integrator)
    adapt_init, adapt_step, adapt_final = base(target_no_refinement_rate)

    def one_step(carry, xs):
        _, rng_key = xs
        state, adaptation_state = carry
        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            info.no_refinement_rate,
        )
        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        init_state = mcmc.walnuts.init(position, logdensity_fn)
        init_adaptation_state = adapt_init(
            position, initial_step_size, inverse_mass_matrix
        )

        if progress_bar:
            print("Running WALNUTS adaptation")
        scan_fn = gen_scan_fn(num_steps, progress_bar=progress_bar)
        keys = jax.random.split(rng_key, num_steps)
        last_state, info = scan_fn(
            one_step,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys),
        )

        last_chain_state, last_adaptation_state = last_state
        step_size, final_inverse_mass_matrix = adapt_final(last_adaptation_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": final_inverse_mass_matrix,
            **extra_parameters,
        }
        return AdaptationResults(last_chain_state, parameters), info

    return AdaptationAlgorithm(run)
