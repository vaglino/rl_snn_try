"""Actor-critic model with v-prop SNN core for Pong."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import lax, random

INJECTIONS_PER_DECISION = 4
PHASES_PER_INJECTION = 3
MICRO_STEPS_PER_DECISION = INJECTIONS_PER_DECISION * PHASES_PER_INJECTION

# Network hyper-parameters (kept modest like the MNIST example)
SNN_INPUT_DIM = 7 * 7 * 16
SNN_HIDDEN = 512
SNN_OUTPUT = 3


@struct.dataclass
class VPropState:
    syn_travel: jnp.ndarray  # (batch, num_edges)
    syn_value: jnp.ndarray   # (batch, num_edges)
    vm: jnp.ndarray          # (batch, n_neurons)
    acc: jnp.ndarray         # (batch, n_outputs)
    phase: jnp.ndarray       # (batch,)


@struct.dataclass
class StepStats:
    phase: jnp.ndarray        # (batch,)
    inject_mask: jnp.ndarray  # (batch,)
    spike_rate: jnp.ndarray   # (batch,)
    input_norm: jnp.ndarray   # (batch,)


@struct.dataclass
class DecisionAux:
    step_phase: jnp.ndarray      # (micro, batch)
    inject_mask: jnp.ndarray     # (micro, batch)
    spike_rate: jnp.ndarray      # (micro, batch)
    logits: jnp.ndarray          # (batch, 3)
    input_norm: jnp.ndarray      # (micro, batch)


# Helper functions adapted from the original script ---------------------------------

def create_directed_small_world(
    n_total: int,
    n_inputs: int,
    n_outputs: int,
    k_nearest: int,
    p_rewire: float,
    p_connect: float,
    key: jax.Array,
) -> Tuple[jax.Array, None, jax.Array]:
    A = jnp.zeros((n_total, n_total), dtype=jnp.int32)
    hidden_start, hidden_end = n_inputs, n_total - n_outputs
    n_hidden = hidden_end - hidden_start
    hidden_slice = slice(hidden_start, hidden_end)
    indices = jnp.arange(hidden_start, hidden_end)
    for i in range(n_hidden):
        A = A.at[indices[i], (i + jnp.arange(1, k_nearest + 1)) % n_hidden + hidden_start].set(1)
    hidden_connections = A[hidden_slice, hidden_slice]
    key, subkey = random.split(key)
    rand_mat = random.uniform(subkey, hidden_connections.shape)
    to_rewire = (rand_mat < p_rewire) & (hidden_connections == 1)
    hidden_connections &= ~to_rewire
    src_rows, _ = jnp.where(to_rewire)
    if num_r := src_rows.shape[0]:
        key, subkey = random.split(key)
        new_cols = random.randint(subkey, (num_r,), 0, n_hidden)
        for s, t in zip(src_rows, new_cols):
            if s != t:
                hidden_connections = hidden_connections.at[s, t].set(1)
    A = A.at[hidden_slice, hidden_slice].set(hidden_connections)
    key, subkey = random.split(key)
    extra = (random.uniform(subkey, (n_hidden, n_hidden)) < p_connect) & ~(hidden_connections.T == 1)
    A = A.at[hidden_slice, hidden_slice].set((hidden_connections | extra).astype(jnp.int32))
    key, subkey = random.split(key)
    A = A.at[:n_inputs, hidden_slice].set(random.uniform(subkey, (n_inputs, n_hidden)) < 2 * p_connect)
    key, subkey = random.split(key)
    A = A.at[hidden_slice, n_total - n_outputs :].set(
        random.uniform(subkey, (n_hidden, n_outputs)) < 2 * p_connect
    )
    if jnp.any(need := (jnp.sum(A[:n_inputs], 1) == 0)):
        key, subkey = random.split(key)
        targets = random.randint(subkey, (n_inputs,), hidden_start, hidden_end)
        for idx in jnp.where(need)[0]:
            A = A.at[idx, targets[idx]].set(1)
    if jnp.any(need := (jnp.sum(A[:, n_total - n_outputs :], 0) == 0)):
        key, subkey = random.split(key)
        sources = random.randint(subkey, (n_outputs,), hidden_start, hidden_end)
        for idx in jnp.where(need)[0]:
            A = A.at[sources[idx], n_total - n_outputs + idx].set(1)
    A = A.at[:, :n_inputs].set(0).at[-n_outputs:, :].set(0)
    return A, None, key


def initialise_edges(adj: jax.Array, key: jax.Array, n_inputs: int):
    key, sub = random.split(key)
    Lmat = adj * random.choice(sub, jnp.arange(3.0, 8.0, 1.0), adj.shape)
    key, sk1, sk2 = random.split(key, 3)
    Wmat = adj * random.uniform(sk1, adj.shape) * ((random.uniform(sk2, adj.shape) > 0.5) * 2 - 1)
    src, tgt = jnp.nonzero(adj)
    L_e = Lmat[src, tgt].at[jnp.where(src < n_inputs)[0]].set(3.0)
    return L_e, Wmat[src, tgt], key


# Modules ----------------------------------------------------------------------------


class ConvTrunk(nnx.Module):
    """Lightweight CNN encoder producing 7x7x16 features (similar scale to MNIST setup)."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(32, 16, kernel_size=(3, 3), strides=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool) -> jax.Array:
        del training
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        return x.reshape(x.shape[0], -1)


class SmallWorldSNN(nnx.Module):
    """Voltage propagation SNN core with preserved original dynamics."""

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        k_nearest: int,
        p_rewire: float,
        p_connect: float,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_neurons = n_hidden + n_outputs
        self.dropout_rate = dropout_rate
        # Original constants
        self.vmax = 1.0
        self.thresh = 0.5
        self.tau = 10.0
        self.dt = 1.0
        key = rngs.params()
        adj, _, key = create_directed_small_world(
            self.n_inputs + self.n_neurons,
            self.n_inputs,
            self.n_outputs,
            k_nearest,
            p_rewire,
            p_connect,
            key,
        )
        src, tgt = jnp.nonzero(adj)
        L_e_init, W_e_init, _ = initialise_edges(adj, key, self.n_inputs)
        input_mask = src < self.n_inputs
        rec_mask = ~input_mask
        input_src = src[input_mask]
        input_tgt = tgt[input_mask] - self.n_inputs
        rec_src = src[rec_mask] - self.n_inputs
        rec_tgt = tgt[rec_mask] - self.n_inputs
        self.src = rec_src
        self.tgt = rec_tgt
        self.num_edges = rec_src.shape[0]
        self.L_e = L_e_init[rec_mask]
        self.W_e = nnx.Param(W_e_init[rec_mask])
        self.input_W = nnx.Param(jnp.zeros((self.n_inputs, self.n_hidden)))
        self.input_W.value = self.input_W.value.at[input_src, input_tgt].set(W_e_init[input_mask])

    def init_state(self, batch_size: int) -> VPropState:
        zeros_edges = jnp.zeros((batch_size, self.num_edges), dtype=jnp.float32)
        zeros_neurons = jnp.zeros((batch_size, self.n_neurons), dtype=jnp.float32)
        zeros_outputs = jnp.zeros((batch_size, self.n_outputs), dtype=jnp.float32)
        phase = jnp.zeros((batch_size,), dtype=jnp.int32)
        return VPropState(
            syn_travel=zeros_edges,
            syn_value=zeros_edges,
            vm=zeros_neurons,
            acc=zeros_outputs,
            phase=phase,
        )

    def encode_current(self, features: jax.Array) -> jax.Array:
        flat = features.reshape(features.shape[0], -1)
        return flat @ self.input_W.value

    def micro_step(self, state: VPropState, input_current: jax.Array, key: jax.Array, dropout_rate: float) -> Tuple[VPropState, StepStats]:
        arrived = jnp.isclose(state.syn_travel, self.L_e, atol=1e-5, rtol=1e-8)
        spikes = (state.syn_value * arrived) * self.W_e.value
        I_syn = jnp.zeros((state.vm.shape[0], self.n_neurons), dtype=state.vm.dtype)
        I_syn = I_syn.at[:, self.tgt].add(spikes)
        inject_mask = (state.phase == 2).astype(input_current.dtype)
        I_syn = I_syn.at[:, : self.n_hidden].add(input_current * inject_mask[:, None])
        input_norm = jnp.linalg.norm(input_current, axis=1) * inject_mask
        vm = state.vm + (-state.vm + I_syn) * (self.dt / self.tau)
        v_exc = jnp.maximum(0.0, vm - self.thresh)
        fired = (v_exc > 0).astype(jnp.bool_)
        # fired = fired.at[:, -self.n_outputs :].set(False)

        def apply_dropout(args):
            fired_in, v_exc_in, key_in = args
            keep = random.bernoulli(key_in, 1.0 - dropout_rate, fired_in.shape)
            fired_mask = fired_in & keep
            scale = keep.astype(v_exc_in.dtype) / (1.0 - dropout_rate)
            return fired_mask, v_exc_in * scale

        fired, v_exc = lax.cond(
            dropout_rate > 0.0,
            apply_dropout,
            lambda args: args[:2],
            (fired, v_exc, key),
        )
        idle = state.syn_travel == 0
        new_spikes = fired[:, self.src] & idle
        syn_travel = state.syn_travel * ~arrived
        syn_value = state.syn_value * ~arrived
        out_vm = vm[:, -self.n_outputs :]
        acc = state.acc + out_vm
        fired_float = fired.astype(vm.dtype)
        vm = vm - vm * fired_float + 0.2 * fired_float
        syn_travel = syn_travel + (syn_travel > 0).astype(vm.dtype) * self.dt * self.vmax
        syn_travel = syn_travel + new_spikes.astype(vm.dtype) * self.dt * self.vmax
        syn_value = syn_value + new_spikes.astype(vm.dtype) * v_exc[:, self.src]
        phase = (state.phase + 1) % 3
        spike_rate = fired_float.mean(axis=1)
        stats = StepStats(phase=state.phase, inject_mask=inject_mask, spike_rate=spike_rate, input_norm=input_norm)
        return VPropState(syn_travel, syn_value, vm, acc, phase), stats


class ActorCriticVProp(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = ConvTrunk(rngs=rngs)
        self.snn = SmallWorldSNN(
            SNN_INPUT_DIM,
            SNN_HIDDEN,
            SNN_OUTPUT,
            k_nearest=16,
            p_rewire=0.3,
            p_connect=0.1,
            dropout_rate=0.0,
            rngs=rngs,
        )
        self.value_head = nnx.Linear(SNN_OUTPUT, 1, rngs=rngs)

    def init_state(self, batch_size: int) -> VPropState:
        return self.snn.init_state(batch_size)

    def _prepare_state(self, state: VPropState) -> VPropState:
        phase = jnp.full_like(state.phase, 2)
        return VPropState(
            syn_travel=state.syn_travel,
            syn_value=state.syn_value,
            vm=state.vm,
            acc=jnp.zeros_like(state.acc),
            phase=phase,
        )

    def decision_unroll(
        self,
        frames: jax.Array,
        state: VPropState,
        *,
        key: jax.Array,
        training: bool,
    ) -> Tuple[jax.Array, jax.Array, VPropState, DecisionAux]:
        if frames.ndim == 4:
            frames = frames[None, ...]
        batch = frames.shape[0]
        state = self._prepare_state(state)
        frame_batch = frames.reshape((batch * INJECTIONS_PER_DECISION, 84, 84, 1))
        encoded = self.encoder(frame_batch, training=training)
        encoded = encoded.reshape((batch, INJECTIONS_PER_DECISION, -1))

        def feature_norm(feat):
            mean = feat.mean(axis=-1, keepdims=True)
            var = jnp.mean((feat - mean) ** 2, axis=-1, keepdims=True)
            return (feat - mean) / jnp.sqrt(var + 1e-5)

        encoded = feature_norm(encoded)
        currents = encoded.reshape((batch * INJECTIONS_PER_DECISION, -1)) @ self.snn.input_W.value
        currents = currents.reshape((batch, INJECTIONS_PER_DECISION, self.snn.n_hidden))
        zero_current = jnp.zeros((batch, self.snn.n_hidden), dtype=currents.dtype)
        sequence = []
        for idx in range(INJECTIONS_PER_DECISION):
            sequence.append(currents[:, idx, :])
            sequence.append(zero_current)
            sequence.append(zero_current)
        micro_currents = jnp.stack(sequence, axis=0)
        keys = random.split(key, MICRO_STEPS_PER_DECISION)
        dropout = self.snn.dropout_rate if training else 0.0

        def body(carry, inputs):
            current, subkey = inputs
            new_state, stats = self.snn.micro_step(carry, current, subkey, dropout)
            return new_state, stats

        next_state, step_stats = lax.scan(body, state, (micro_currents, keys))
        avg_acc = next_state.acc / MICRO_STEPS_PER_DECISION
        logits = avg_acc - avg_acc.mean(axis=1, keepdims=True)
        value = self.value_head(avg_acc)
        reset_state = VPropState(
            syn_travel=next_state.syn_travel,
            syn_value=next_state.syn_value,
            vm=next_state.vm,
            acc=jnp.zeros_like(next_state.acc),
            phase=next_state.phase,
        )
        aux = DecisionAux(
            step_phase=step_stats.phase,
            inject_mask=step_stats.inject_mask,
            spike_rate=step_stats.spike_rate,
            logits=logits,
            input_norm=step_stats.input_norm,
        )
        return logits, value, reset_state, aux
