"""Basic sanity tests for the v-prop actor-critic implementation."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
from flax import nnx

from model import ActorCriticVProp, MICRO_STEPS_PER_DECISION
from train_online_ac import build_train_step

MICRO_PHASE_SEQUENCE = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]


def run_schedule_test():
    rngs = nnx.Rngs(0)
    agent = ActorCriticVProp(rngs=rngs)
    state = agent.init_state(1)
    frames = jnp.stack(
        [jnp.full((84, 84, 1), fill_value=float(i + 1)) for i in range(4)], axis=0
    )
    key = jax.random.PRNGKey(0)
    logits, value, next_state, aux = agent.decision_unroll(frames, state, key=key, training=True)
    phase = np.array(aux.step_phase[:, 0])
    assert len(phase) == MICRO_STEPS_PER_DECISION
    assert phase.tolist() == MICRO_PHASE_SEQUENCE, f"Phase sequence mismatch: {phase}"
    input_norm = np.array(aux.input_norm[:, 0])
    inject_steps = np.where(input_norm > 1e-6)[0].tolist()
    assert inject_steps == [0, 3, 6, 9], f"Injection steps incorrect: {inject_steps}"
    assert logits.shape == (1, 3)
    assert value.shape == (1, 1)
    assert next_state.vm.shape[0] == 1
    assert np.allclose(np.array(aux.logits), np.array(logits))


def run_gradient_test():
    rngs = nnx.Rngs(1)
    agent = ActorCriticVProp(rngs=rngs)
    graphdef, params, static = nnx.split(agent, nnx.Param, ...)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    opt_state = optimizer.init(params)
    train_step = build_train_step(graphdef, static, optimizer)

    rng = jax.random.PRNGKey(123)
    init_state = agent.init_state(1)
    init_state = tree_util.tree_map(jnp.asarray, init_state)
    frames = jnp.ones((16, 4, 84, 84, 1)) * 0.5
    actions = jnp.zeros((16,), dtype=jnp.int32)
    rewards = jnp.ones((16,), dtype=jnp.float32)
    dones = jnp.zeros((16,), dtype=jnp.float32)
    bootstrap = jnp.array(0.0, dtype=jnp.float32)

    params2, opt_state2, metrics = train_step(
        params,
        opt_state,
        init_state,
        frames,
        actions,
        rewards,
        dones,
        bootstrap,
        rng,
    )
    grad_norm = float(metrics["grad_norm"])
    assert grad_norm > 0.0, "Gradient norm should be non-zero"


def main():
    run_schedule_test()
    run_gradient_test()
    print("Sanity tests passed.")


if __name__ == "__main__":
    main()
