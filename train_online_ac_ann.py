"""Online actor-critic training using a feed-forward ANN baseline."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Deque, Dict, List
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_wrappers import make_pong_env, repeat_action_four
from ann_model import ActorCriticANN, FRAMES_PER_DECISION

WINDOW_W = 1
LEARN_TAIL_K = 1
GAMMA_DEC = 0.99
GAE_LAMBDA = 0.95
LR = 5e-4
ENTROPY_BETA = 0.05
VALUE_COEF = 0.5
GRAD_CLIP = 10.0
SEED = 42
MAX_STEPS = 10_000
BEHAVIOR_EPS = 0.1
ACTION_MAP = np.array([0, 2, 3], dtype=np.int32)
VIDEO_DIR = "videos_ann"
DEBUG_DIR = "debug_ann"
DEBUG_SAVE_STEPS = 32
DEBUG_CSV = os.path.join(DEBUG_DIR, "decision_metrics.csv")
FIRE_ACTION = 1  # ALE action id for FIRE


@dataclass
class Transition:
    frames: np.ndarray  # (4,84,84,1)
    action: int
    reward: float
    done: bool
    value: float
    bootstrap: float = float("nan")


def behaviour_distribution(logits: jax.Array) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=-1)
    return (1.0 - BEHAVIOR_EPS) * probs + BEHAVIOR_EPS / logits.shape[-1]


def build_train_step(
    graphdef: nnx.GraphDef,
    static: nnx.State,
    optimizer: optax.GradientTransformation,
):
    def loss_on_window(
        params: nnx.State,
        frames: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        bootstrap_value: jax.Array,
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        module = nnx.merge(graphdef, params, static)
        logits, values = module(frames, training=True)
        values = values.squeeze(-1)
        behaviour = behaviour_distribution(logits)
        log_probs = jnp.log(behaviour + 1e-8)
        action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)
        entropy = -(behaviour * log_probs).sum(axis=-1)
        bootstrap = jax.lax.stop_gradient(bootstrap_value)
        next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)
        not_done = 1.0 - dones
        deltas = rewards + GAMMA_DEC * next_values * not_done - values

        def gae_scan(carry, inputs):
            delta, mask = inputs
            adv = delta + GAMMA_DEC * GAE_LAMBDA * mask * carry
            return adv, adv

        _, advantages_rev = jax.lax.scan(
            gae_scan,
            jnp.zeros_like(deltas[-1]),
            (deltas[::-1], not_done[::-1]),
        )
        advantages = advantages_rev[::-1]
        returns = advantages + values
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)
        mask = jnp.ones_like(rewards)
        normalizer = mask.sum() + 1e-8
        policy_loss = -(mask * action_log_probs * advantages).sum() / normalizer
        value_loss = 0.5 * (mask * (returns - values) ** 2).sum() / normalizer
        entropy_loss = -(mask * entropy).sum() / normalizer
        loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_BETA * entropy_loss
        action_ids = jnp.arange(logits.shape[-1])
        counts = jnp.array([(actions == a).sum() for a in range(logits.shape[-1])], dtype=advantages.dtype)
        sums = jnp.array([((advantages) * (actions == a)).sum() for a in range(logits.shape[-1])], dtype=advantages.dtype)
        mean_adv = jnp.where(counts > 0, sums / (counts + 1e-8), 0.0)
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
            "adv_mean_action0": mean_adv[0],
            "adv_mean_action1": mean_adv[1],
            "adv_mean_action2": mean_adv[2],
        }
        return loss, metrics

    @jax.jit
    def train_step(
        params: nnx.State,
        opt_state: optax.OptState,
        frames: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        bootstrap_value: jax.Array,
    ):
        def loss_fn(p):
            return loss_on_window(p, frames, actions, rewards, dones, bootstrap_value)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        metrics = {**metrics, "loss": loss, "grad_norm": grad_norm}
        return new_params, new_opt_state, metrics

    return train_step


def make_numpy_transition(frames: List[np.ndarray], action: int, reward: float, done: bool, value: float) -> Transition:
    stacked = np.stack(frames, axis=0).astype(np.float32)
    return Transition(stacked, action, reward, done, float(value))


def main() -> None:
    rng = jax.random.PRNGKey(SEED)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_records: List[Dict[str, float]] = []
    with open(DEBUG_CSV, "w") as f:
        f.write("step,action,env_action,reward,value,logit0,logit1,logit2,prob0,prob1,prob2,frame_diff\n")
    env = make_pong_env(SEED, record_dir=VIDEO_DIR)
    rngs = nnx.Rngs(SEED)
    model = ActorCriticANN(rngs=rngs)
    graphdef, params, static = nnx.split(model, nnx.Param, ...)
    optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR, eps=1e-5))
    opt_state = optimizer.init(params)
    train_step = build_train_step(graphdef, static, optimizer)

    action = 0
    env.reset(seed=SEED)
    env.step(FIRE_ACTION)
    transitions: Deque[Transition] = deque(maxlen=WINDOW_W + 1)
    episode_return = 0.0
    episode_length = 0
    episode = 0
    start_time = time.time()

    for step in range(MAX_STEPS):
        env_action = int(ACTION_MAP[action])
        frames, sum_reward, done, _ = repeat_action_four(env, env_action)
        frames_np = [np.array(f, dtype=np.float32) for f in frames]
        stacked_frames = np.stack(frames_np, axis=0)
        frames_jnp = jnp.asarray(stacked_frames)
        rng, call_key, sample_key = jax.random.split(rng, 3)
        module = nnx.merge(graphdef, params, static)
        logits, value = module(frames_jnp[None, ...], training=True)
        logits = logits[0]
        value_scalar = float(value.squeeze())
        behaviour = behaviour_distribution(logits)
        action = int(jax.random.choice(sample_key, behaviour.shape[-1], p=behaviour))

        if episode_length < DEBUG_SAVE_STEPS:
            diff = float(np.mean(np.abs(stacked_frames[1:] - stacked_frames[:-1])))
            np.savez(
                os.path.join(DEBUG_DIR, f"decision_{step:05d}.npz"),
                frames=stacked_frames,
                logits=np.array(logits),
                behaviour=np.array(behaviour),
                action=action,
                env_action=int(ACTION_MAP[action]),
                reward=sum_reward,
                done=done,
            )
            record = {
                "step": step,
                "action": action,
                "env_action": int(ACTION_MAP[action]),
                "reward": sum_reward,
                "value": value_scalar,
                "logit0": float(logits[0]),
                "logit1": float(logits[1]),
                "logit2": float(logits[2]),
                "prob0": float(behaviour[0]),
                "prob1": float(behaviour[1]),
                "prob2": float(behaviour[2]),
                "frame_diff": diff,
            }
            debug_records.append(record)
            with open(DEBUG_CSV, "a") as f:
                f.write(
                    f"{record['step']},{record['action']},{record['env_action']},{record['reward']},{record['value']},{record['logit0']},{record['logit1']},{record['logit2']},{record['prob0']},{record['prob1']},{record['prob2']},{record['frame_diff']}\n"
                )

        transition = make_numpy_transition(frames_np, action, sum_reward, done, value_scalar)
        transitions.append(transition)
        if len(transitions) >= 2 and not transitions[-2].done:
            transitions[-2].bootstrap = value_scalar

        episode_return += sum_reward
        episode_length += 1

        if done:
            transitions[-1].bootstrap = 0.0
            action = 0
            episode += 1
            env.reset()
            env.step(FIRE_ACTION)
            print(
                f"Episode {episode}: return={episode_return:.1f} length={episode_length} decisions, total_steps={step+1} | video -> {VIDEO_DIR}"
            )
            if debug_records:
                steps_plot = [r["step"] for r in debug_records]
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                axes[0].plot(steps_plot, [r["reward"] for r in debug_records], label="reward")
                axes[0].plot(steps_plot, [r["value"] for r in debug_records], label="value")
                axes[0].set_ylabel("Reward/Value")
                axes[0].legend()

                axes[1].plot(steps_plot, [r["prob0"] for r in debug_records], label="prob0")
                axes[1].plot(steps_plot, [r["prob1"] for r in debug_records], label="prob1")
                axes[1].plot(steps_plot, [r["prob2"] for r in debug_records], label="prob2")
                axes[1].set_xlabel("Decision step")
                axes[1].set_ylabel("Behaviour prob")
                axes[1].legend()

                fig.tight_layout()
                fig.savefig(os.path.join(DEBUG_DIR, f"decision_metrics_ep{episode:03d}.png"))
                plt.close(fig)
                debug_records.clear()
            episode_return = 0.0
            episode_length = 0

        if len(transitions) >= LEARN_TAIL_K + 1:
            tail = list(transitions)[-LEARN_TAIL_K - 1 : -1]
            if np.isnan(tail[-1].bootstrap):
                continue
            frames_batch = jnp.asarray(np.stack([t.frames for t in tail], axis=0))
            actions_batch = jnp.array([t.action for t in tail], dtype=jnp.int32)
            rewards_batch = jnp.array([t.reward for t in tail], dtype=jnp.float32)
            dones_batch = jnp.array([t.done for t in tail], dtype=jnp.float32)
            bootstrap_value = jnp.array(tail[-1].bootstrap, dtype=jnp.float32)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                frames_batch,
                actions_batch,
                rewards_batch,
                dones_batch,
                bootstrap_value,
            )
            if step % 50 == 0:
                metrics_np = {k: float(v) for k, v in metrics.items()}
                elapsed = time.time() - start_time
                print(f"step={step} loss={metrics_np['loss']:.4f} entropy={metrics_np['entropy']:.3f} time={elapsed:.1f}s")

    print("Training completed")
    env.close()

    # per-episode plots saved during training


if __name__ == "__main__":
    main()
