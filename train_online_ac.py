"""Online actor-critic training with truncated BPTT for Pong."""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import tree_util
from flax import nnx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_wrappers import make_pong_env, repeat_action_four
from model import (
    ActorCriticVProp,
    VPropState,
    MICRO_STEPS_PER_DECISION,
    INJECTIONS_PER_DECISION,
)

WINDOW_W = 32
LEARN_TAIL_K = 32
GAMMA_DEC = 0.99
GAE_LAMBDA = 0.95
LR = 1e-4
ENTROPY_BETA = 0.03
VALUE_COEF = 0.5
GRAD_CLIP = 10.0
SEED = 42
MAX_STEPS = 2_000

# Map policy action indices (0: stay, 1: up, 2: down) to environment action IDs.
ACTION_MAP = np.array([0, 2, 3], dtype=np.int32)
VIDEO_DIR = "videos"
BEHAVIOR_EPS = 0.1
FIRE_ACTION = 1
DEBUG_DIR = "debug_snn"
DEBUG_SAVE_STEPS = 32
DEBUG_CSV = os.path.join(DEBUG_DIR, "decision_metrics.csv")


@dataclass
class Transition:
    frames: np.ndarray
    action: int
    reward: float
    done: bool
    value: float
    bootstrap: float = float("nan")


def build_train_step(
    graphdef: nnx.GraphDef,
    static: nnx.State,
    optimizer: optax.GradientTransformation,
):
    def loss_on_window(
        params: nnx.State,
        init_state: VPropState,
        frames: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        bootstrap_value: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        module = nnx.merge(graphdef, params, static)
        tail_len = frames.shape[0]

        def scan_body(carry, inputs):
            state, rng = carry
            frame = inputs
            rng, sub = jax.random.split(rng)
            logits, value, next_state, _ = module.decision_unroll(frame, state, key=sub, training=True)
            logits = logits.squeeze(0)
            value = value.squeeze()
            return (next_state, rng), (logits, value)

        init_state = tree_util.tree_map(jax.lax.stop_gradient, init_state)
        (_, _), (logits_seq, values_seq) = jax.lax.scan(
            scan_body,
            (init_state, key),
            frames,
        )
        # Append bootstrap for final decision
        bootstrap = jax.lax.stop_gradient(bootstrap_value)
        next_values = jnp.concatenate([values_seq[1:], bootstrap[None]], axis=0)
        dones_float = dones.astype(jnp.float32)
        not_done = 1.0 - dones_float
        deltas = rewards + GAMMA_DEC * next_values * not_done - values_seq

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
        returns = advantages + values_seq
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)
        softmax_probs = jax.nn.softmax(logits_seq, axis=-1)
        pi = (1.0 - BEHAVIOR_EPS) * softmax_probs + BEHAVIOR_EPS / logits_seq.shape[-1]
        log_probs = jnp.log(pi + 1e-8)
        action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)
        entropy = -(pi * log_probs).sum(axis=-1)
        mask = jnp.ones((tail_len,), dtype=jnp.float32)
        normalizer = mask.sum() + 1e-8
        policy_loss = -(mask * action_log_probs * advantages).sum() / normalizer
        value_loss = 0.5 * (mask * (returns - values_seq) ** 2).sum() / normalizer
        entropy_loss = -(mask * entropy).sum() / normalizer
        loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_BETA * entropy_loss
        counts = jnp.array([(actions == a).sum() for a in range(logits_seq.shape[-1])], dtype=advantages.dtype)
        sums = jnp.array([((advantages) * (actions == a)).sum() for a in range(logits_seq.shape[-1])], dtype=advantages.dtype)
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
        init_state: VPropState,
        frames: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        bootstrap_value: jax.Array,
        key: jax.Array,
    ):
        def loss_fn(p):
            return loss_on_window(p, init_state, frames, actions, rewards, dones, bootstrap_value, key)

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


def main():
    rng = jax.random.PRNGKey(SEED)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_records: List[Dict[str, float]] = []
    last_metrics = {
        "adv_mean_action0": 0.0,
        "adv_mean_action1": 0.0,
        "adv_mean_action2": 0.0,
    }
    with open(DEBUG_CSV, "w") as f:
        f.write(
            "step,action,env_action,sampled_next,reward,value,logit0,logit1,logit2,prob0,prob1,prob2,adv0,adv1,adv2,frame_diff,input_norm,spike_rate\n"
        )
    env = make_pong_env(SEED, record_dir=VIDEO_DIR)
    rngs = nnx.Rngs(SEED)
    agent = ActorCriticVProp(rngs=rngs)
    graphdef, params, static = nnx.split(agent, nnx.Param, ...)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(LR, eps=1e-5),
    )
    opt_state = optimizer.init(params)
    train_step = build_train_step(graphdef, static, optimizer)

    current_state = agent.init_state(1)
    state_history: Deque[VPropState] = deque(maxlen=WINDOW_W + 2)
    state_history.append(current_state)
    action = 0
    env.reset(seed=SEED)
    env.step(FIRE_ACTION)
    transitions: Deque[Transition] = deque(maxlen=WINDOW_W + 1)
    episode_return = 0.0
    episode_length = 0
    episode = 0

    start_time = time.time()
    for step in range(MAX_STEPS):
        action_to_execute = action
        env_action = int(ACTION_MAP[action_to_execute])
        frames, sum_reward, done, info = repeat_action_four(env, env_action)
        frames_np = [np.array(f, dtype=np.float32) for f in frames]
        stacked_frames = np.stack(frames_np, axis=0)
        frames_jnp = jnp.asarray(stacked_frames)
        rng, call_key, sample_key = jax.random.split(rng, 3)
        module = nnx.merge(graphdef, params, static)
        logits, value, next_state, aux = module.decision_unroll(frames_jnp, current_state, key=call_key, training=True)
        value_scalar = float(value.squeeze())
        probs = jax.nn.softmax(logits[0])
        mixed_probs = (1.0 - BEHAVIOR_EPS) * probs + BEHAVIOR_EPS / 3.0
        next_action = int(jax.random.choice(sample_key, 3, p=mixed_probs))

        if episode_length < DEBUG_SAVE_STEPS:
            diff = float(np.mean(np.abs(stacked_frames[1:] - stacked_frames[:-1])))
            np.savez(
                os.path.join(DEBUG_DIR, f"decision_{step:05d}.npz"),
                frames=stacked_frames,
                logits=np.array(logits[0]),
                behaviour=np.array(mixed_probs),
                action_executed=action_to_execute,
                env_action=int(env_action),
                sampled_next=next_action,
                reward=sum_reward,
                value=value_scalar,
                done=done,
                avg_acc=np.array(aux.logits[0]),
                step_phase=np.array(aux.step_phase),
                inject_mask=np.array(aux.inject_mask),
                spike_rate=np.array(aux.spike_rate),
                input_norm=np.array(aux.input_norm),
            )
            record = {
                    "step": step,
                    "action": action_to_execute,
                    "env_action": int(env_action),
                    "sampled_next": next_action,
                    "reward": sum_reward,
                    "value": value_scalar,
                    "logit0": float(logits[0][0]),
                    "logit1": float(logits[0][1]),
                    "logit2": float(logits[0][2]),
                    "prob0": float(mixed_probs[0]),
                    "prob1": float(mixed_probs[1]),
                    "prob2": float(mixed_probs[2]),
                    "adv0": float(last_metrics["adv_mean_action0"]),
                    "adv1": float(last_metrics["adv_mean_action1"]),
                    "adv2": float(last_metrics["adv_mean_action2"]),
                    "frame_diff": diff,
                    "input_norm": float(aux.input_norm.sum()),
                    "spike_rate": float(aux.spike_rate.mean()),
                }
            debug_records.append(record)
            with open(DEBUG_CSV, "a") as f:
                f.write(
                    f"{record['step']},{record['action']},{record['env_action']},{record['sampled_next']},{record['reward']},{record['value']},{record['logit0']},{record['logit1']},{record['logit2']},{record['prob0']},{record['prob1']},{record['prob2']},{record['adv0']},{record['adv1']},{record['adv2']},{record['frame_diff']},{record['input_norm']},{record['spike_rate']}\n"
                )

        transition = make_numpy_transition(frames_np, action_to_execute, sum_reward, done, value_scalar)
        transitions.append(transition)
        if len(transitions) >= 2 and not transitions[-2].done:
            transitions[-2].bootstrap = value_scalar
        if done:
            next_state = agent.init_state(1)
        state_history.append(next_state)
        episode_return += sum_reward
        episode_length += 1

        if done:
            transitions[-1].bootstrap = 0.0
            current_state = next_state
            action = 0
            episode += 1
            env.reset()
            env.step(FIRE_ACTION)
            print(
                f"Episode {episode}: return={episode_return:.1f} length={episode_length} decisions, total_steps={step+1}"
            )
            if debug_records:
                steps_plot = [r["step"] for r in debug_records]
                fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
                axes[0].plot(steps_plot, [r["reward"] for r in debug_records], label="reward")
                axes[0].plot(steps_plot, [r["value"] for r in debug_records], label="value")
                axes[0].set_ylabel("Reward/Value")
                axes[0].legend()

                axes[1].plot(steps_plot, [r["logit0"] for r in debug_records], label="logit0")
                axes[1].plot(steps_plot, [r["logit1"] for r in debug_records], label="logit1")
                axes[1].plot(steps_plot, [r["logit2"] for r in debug_records], label="logit2")
                axes[1].set_ylabel("Logits")
                axes[1].legend()

                axes[2].plot(steps_plot, [r["adv0"] for r in debug_records], label="adv0")
                axes[2].plot(steps_plot, [r["adv1"] for r in debug_records], label="adv1")
                axes[2].plot(steps_plot, [r["adv2"] for r in debug_records], label="adv2")
                axes[2].set_xlabel("Decision step")
                axes[2].set_ylabel("Mean advantage")
                axes[2].legend()

                fig.tight_layout()
                fig.savefig(os.path.join(DEBUG_DIR, f"decision_metrics_ep{episode:03d}.png"))
                plt.close(fig)
                debug_records.clear()
                last_metrics = {
                    "adv_mean_action0": 0.0,
                    "adv_mean_action1": 0.0,
                    "adv_mean_action2": 0.0,
                }
            episode_return = 0.0
            episode_length = 0
        else:
            current_state = next_state
            action = next_action

        if len(transitions) >= LEARN_TAIL_K + 1:
            tail = list(transitions)[-LEARN_TAIL_K - 1 : -1]
            if np.isnan(tail[-1].bootstrap):
                continue
            init_state = list(state_history)[-LEARN_TAIL_K - 1]
            init_state = tree_util.tree_map(jnp.asarray, init_state)
            frames_batch = jnp.asarray(np.stack([t.frames for t in tail], axis=0))
            actions_batch = jnp.array([t.action for t in tail], dtype=jnp.int32)
            rewards_batch = jnp.array([t.reward for t in tail], dtype=jnp.float32)
            dones_batch = jnp.array([t.done for t in tail], dtype=jnp.float32)
            bootstrap_value = jnp.array(tail[-1].bootstrap, dtype=jnp.float32)
            rng, loss_key = jax.random.split(rng)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                init_state,
                frames_batch,
                actions_batch,
                rewards_batch,
                dones_batch,
                bootstrap_value,
                loss_key,
            )
            last_metrics = {
                "adv_mean_action0": float(metrics.get("adv_mean_action0", 0.0)),
                "adv_mean_action1": float(metrics.get("adv_mean_action1", 0.0)),
                "adv_mean_action2": float(metrics.get("adv_mean_action2", 0.0)),
            }
            if step % 50 == 0:
                metrics_np = {k: float(v) for k, v in metrics.items()}
                elapsed = time.time() - start_time
                print(f"step={step} loss={metrics_np['loss']:.4f} entropy={metrics_np['entropy']:.3f} time={elapsed:.1f}s")

    print("Training completed")
    env.close()

    # no final aggregate plot; per-episode plots already saved


if __name__ == "__main__":
    main()
