"""Tests for the WALNUTS kernel."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.walnuts as walnuts
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import IntegratorState
from tests.fixtures import BlackJAXTest, std_normal_logdensity


def _normal_components(position, momentum):
    metric = metrics.gaussian_euclidean(jnp.ones(position.size))
    logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
    integrator = integrators.velocity_verlet(logdensity_fn, metric.kinetic_energy)
    state = IntegratorState(
        position,
        momentum,
        logdensity_fn(position),
        jax.grad(logdensity_fn)(position),
    )
    return logdensity_fn, metric, integrator, state


class WALNUTSTest(BlackJAXTest):
    def test_micro_step_refines_large_macro_step(self):
        _, metric, integrator, state = _normal_components(
            jnp.array([1.0]), jnp.array([1.0])
        )

        info = jax.jit(
            lambda state: walnuts.micro_step(
                integrator,
                metric.kinetic_energy,
                state,
                1.0,
                0.01,
                4,
            )
        )(state)

        self.assertGreater(int(info.level), 0)
        self.assertGreater(int(info.num_integration_steps), 1)

    def test_micro_step_pmf(self):
        self.assertAlmostEqual(
            float(walnuts.micro_step_pmf(2, 2, 4, "r2p")), 2.0 / 3.0
        )
        self.assertAlmostEqual(
            float(walnuts.micro_step_pmf(3, 2, 4, "r2p")), 1.0 / 3.0
        )
        self.assertEqual(float(walnuts.micro_step_pmf(4, 4, 4, "r2p")), 1.0)
        self.assertEqual(
            float(walnuts.micro_step_pmf(3, 2, 4, "deterministic")), 0.0
        )

    def test_leaf_weight_finite_or_negative_infinity(self):
        _, metric, integrator, state = _normal_components(
            jnp.array([1.0]), jnp.array([1.0])
        )
        initial_log_weight = -walnuts._energy(metric.kinetic_energy, state)

        _, finite_log_weight, is_divergent, _ = walnuts._build_leaf(
            self.next_key(),
            integrator,
            metric.kinetic_energy,
            state,
            initial_log_weight,
            initial_log_weight,
            1,
            0.1,
            0.5,
            3,
            "r2p",
            1000.0,
        )
        self.assertTrue(bool(jnp.isfinite(finite_log_weight)))
        self.assertFalse(bool(is_divergent))

        _, rejected_log_weight, is_divergent, _ = walnuts._build_leaf(
            self.next_key(),
            integrator,
            metric.kinetic_energy,
            state,
            initial_log_weight,
            initial_log_weight,
            1,
            0.1,
            0.5,
            3,
            "r2p",
            0.0,
        )
        self.assertTrue(bool(jnp.isneginf(rejected_log_weight)))
        self.assertTrue(bool(is_divergent))

    def test_subtree_builder_is_jittable(self):
        _, metric, integrator, state = _normal_components(
            jnp.array([0.0]), jnp.array([0.5])
        )
        initial_log_weight = -walnuts._energy(metric.kinetic_energy, state)

        build = jax.jit(
            lambda key: walnuts._build_subtree(
                key,
                integrator,
                metric.kinetic_energy,
                metric,
                state,
                initial_log_weight,
                initial_log_weight,
                1,
                2,
                0.1,
                0.5,
                3,
                2,
                "r2p",
                1000.0,
            )
        )
        result = build(self.next_key())

        self.assertGreaterEqual(int(result.diagnostics.num_macro_steps), 1)
        self.assertTrue(bool(jnp.isfinite(result.subtree.log_weight)))

    def test_sampling_algorithm_interface(self):
        sampler = blackjax.walnuts(
            std_normal_logdensity,
            step_size=0.2,
            inverse_mass_matrix=jnp.array([1.0]),
            max_num_doublings=2,
            max_num_micro_doublings=2,
        )
        state = sampler.init(jnp.array(0.5))
        self.assertIsInstance(state, HMCState)

        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, HMCState)
        self.assertIsInstance(info, walnuts.WALNUTSInfo)
        self.assertGreaterEqual(int(info.num_macro_steps), 2)

    def test_correct_sampling_on_standard_normal(self):
        sampler = blackjax.walnuts(
            std_normal_logdensity,
            step_size=0.5,
            inverse_mass_matrix=jnp.array([1.0]),
            energy_threshold=0.5,
            max_num_doublings=4,
            max_num_micro_doublings=3,
        )
        state = sampler.init(jnp.array(0.0))
        step = jax.jit(sampler.step)
        keys = jax.random.split(self.next_key(), 800)

        def one_step(state, key):
            state, _ = step(key, state)
            return state, state.position

        _, samples = jax.lax.scan(one_step, state, keys)
        samples = samples[200:]

        self.assertAlmostEqual(float(jnp.mean(samples)), 0.0, delta=0.35)
        self.assertAlmostEqual(float(jnp.std(samples)), 1.0, delta=0.35)

    def test_pytree_position(self):
        sampler = blackjax.walnuts(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(2),
            max_num_doublings=2,
            max_num_micro_doublings=2,
        )
        state = sampler.init({"a": jnp.array(0.0), "b": jnp.array(1.0)})
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)

        self.assertIn("a", new_state.position)
        self.assertIn("b", new_state.position)
        self.assertGreaterEqual(float(info.no_refinement_rate), 0.0)

    def test_deterministic_micro_distribution_smoke(self):
        sampler = blackjax.walnuts(
            std_normal_logdensity,
            step_size=0.2,
            inverse_mass_matrix=jnp.array([1.0]),
            max_num_doublings=2,
            max_num_micro_doublings=2,
            micro_step_distribution="deterministic",
        )
        state = sampler.init(jnp.array(0.0))
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)

        self.assertTrue(bool(jnp.isfinite(new_state.logdensity)))
        self.assertGreaterEqual(int(info.max_micro_steps), 1)

    def test_walnuts_adaptation_returns_parameters(self):
        warmup = blackjax.walnuts_adaptation(
            std_normal_logdensity,
            initial_step_size=0.05,
            target_no_refinement_rate=0.5,
            energy_threshold=0.5,
            max_num_doublings=2,
            max_num_micro_doublings=2,
        )

        (_, parameters), info = warmup.run(
            self.next_key(), jnp.array([0.0]), num_steps=20
        )

        self.assertGreater(float(parameters["step_size"]), 0.05)
        np.testing.assert_allclose(parameters["energy_threshold"], 0.5)
        self.assertGreaterEqual(float(info.info.no_refinement_rate[-1]), 0.0)

    def test_walnuts_adaptation_preserves_float64(self):
        previous_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            inverse_mass_matrix = jnp.ones(1, dtype=jnp.float64)
            warmup = blackjax.walnuts_adaptation(
                std_normal_logdensity,
                inverse_mass_matrix=inverse_mass_matrix,
                initial_step_size=jnp.asarray(0.05, dtype=jnp.float64),
                target_no_refinement_rate=0.5,
                energy_threshold=jnp.asarray(0.5, dtype=jnp.float64),
                max_num_doublings=2,
                max_num_micro_doublings=2,
            )

            (_, parameters), info = warmup.run(
                self.next_key(),
                jnp.array([0.0], dtype=jnp.float64),
                num_steps=5,
            )

            self.assertEqual(parameters["step_size"].dtype, jnp.float64)
            self.assertEqual(info.info.no_refinement_rate.dtype, jnp.float64)
            sampler = blackjax.walnuts(std_normal_logdensity, **parameters)
            state = sampler.init(jnp.array([0.0], dtype=jnp.float64))
            new_state, sampler_info = jax.jit(sampler.step)(self.next_key(), state)

            self.assertEqual(new_state.position.dtype, jnp.float64)
            self.assertEqual(sampler_info.no_refinement_rate.dtype, jnp.float64)
        finally:
            jax.config.update("jax_enable_x64", previous_x64)


if __name__ == "__main__":
    chex.set_n_cpu_devices(1)
    absltest.main()
