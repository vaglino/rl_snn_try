"""Feed-forward CNN + MLP actor-critic for Pong (baseline without SNN)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

FRAMES_PER_DECISION = 4
FEATURE_DIM = 7 * 7 * 16
HIDDEN_DIM = 256
NUM_ACTIONS = 3


class ConvTrunk(nnx.Module):
    """Matches the lightweight conv encoder used with the SNN."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(32, 16, kernel_size=(3, 3), strides=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool) -> jax.Array:
        del training  # no training-specific behaviour
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        return x.reshape(x.shape[0], -1)


class ActorCriticANN(nnx.Module):
    """CNN + simple recurrent aggregator over the four sequential frames."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = ConvTrunk(rngs=rngs)
        self.frame_proj = nnx.Linear(FEATURE_DIM, HIDDEN_DIM, rngs=rngs)
        self.recurrent = nnx.Linear(HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.policy_head = nnx.Linear(HIDDEN_DIM, NUM_ACTIONS, rngs=rngs)
        self.value_head = nnx.Linear(HIDDEN_DIM, 1, rngs=rngs)

    def __call__(self, frames: jax.Array, *, training: bool) -> tuple[jax.Array, jax.Array]:
        """frames: (batch,4,84,84,1) -> logits,value with frame-by-frame aggregation."""
        if frames.ndim == 4:
            frames = frames[None, ...]
        batch = frames.shape[0]
        flat = frames.reshape((batch * FRAMES_PER_DECISION, 84, 84, 1))
        feats = self.encoder(flat, training=training)
        feats = feats.reshape((batch, FRAMES_PER_DECISION, FEATURE_DIM))

        def step(carry, x):
            h = carry
            x_proj = nnx.relu(self.frame_proj(x))
            h_proj = self.recurrent(h)
            new_h = nnx.relu(x_proj + h_proj)
            return new_h, new_h

        init_h = jnp.zeros((batch, HIDDEN_DIM), dtype=feats.dtype)
        final_h, _ = jax.lax.scan(step, init_h, feats.swapaxes(0, 1))
        hidden = final_h
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value
