import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial
import optax
from flax import nnx
import time
from jax import checkpoint

# pip install keras_core
from keras.datasets import mnist
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize

# ——————————————————————————————————————————————————————————————
# Global constants and Helpers (unchanged)
# ——————————————————————————————————————————————————————————————
thresh, dt, ds, t_max, tau, vmax = 0.5, 1.0, 1.0, 30, 10.0, 1.0

def create_directed_small_world(n_total, n_inputs, n_outputs, k_nearest, p_rewire, p_connect, key):
    A = jnp.zeros((n_total, n_total), dtype=jnp.int32)
    hidden_start, hidden_end = n_inputs, n_total - n_outputs
    n_hidden = hidden_end - hidden_start; hidden_slice = slice(hidden_start, hidden_end)
    indices = jnp.arange(hidden_start, hidden_end)
    for i in range(n_hidden):
        A = A.at[indices[i], (i + jnp.arange(1, k_nearest + 1)) % n_hidden + hidden_start].set(1)
    hidden_connections = A[hidden_slice, hidden_slice]
    key, subkey = random.split(key); rand_mat = random.uniform(subkey, hidden_connections.shape)
    to_rewire = (rand_mat < p_rewire) & (hidden_connections == 1)
    hidden_connections &= ~to_rewire
    src_rows, _ = jnp.where(to_rewire)
    if num_r := src_rows.shape[0]:
        key, subkey = random.split(key); new_cols = random.randint(subkey, (num_r,), 0, n_hidden)
        for s, t in zip(src_rows, new_cols):
            if s != t: hidden_connections = hidden_connections.at[s, t].set(1)
    A = A.at[hidden_slice, hidden_slice].set(hidden_connections)
    key, subkey = random.split(key); extra = (random.uniform(subkey, (n_hidden, n_hidden)) < p_connect) & ~(hidden_connections.T == 1)
    A = A.at[hidden_slice, hidden_slice].set((hidden_connections | extra).astype(jnp.int32))
    key, subkey = random.split(key); A = A.at[:n_inputs, hidden_slice].set(random.uniform(subkey, (n_inputs, n_hidden)) < 2*p_connect)
    key, subkey = random.split(key); A = A.at[hidden_slice, n_total-n_outputs:].set(random.uniform(subkey, (n_hidden, n_outputs)) < 2*p_connect)
    if jnp.any(need := (jnp.sum(A[:n_inputs], 1) == 0)):
        key, subkey = random.split(key); t = random.randint(subkey, (n_inputs,), hidden_start, hidden_end)
        for i in jnp.where(need)[0]: A = A.at[i, t[i]].set(1)
    if jnp.any(need := (jnp.sum(A[:, n_total-n_outputs:], 0) == 0)):
        key, subkey = random.split(key); s = random.randint(subkey, (n_outputs,), hidden_start, hidden_end)
        for i in jnp.where(need)[0]: A = A.at[s[i], n_total-n_outputs+i].set(1)
    A = A.at[:, :n_inputs].set(0).at[-n_outputs:, :].set(0)
    return A, None, key

def load_mnist_data(train_size=0.8, val_size=0.1, resample_fraction=0.2, target_size=(14,14), random_state=42):
    (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
    X, y = np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te])
    X = StandardScaler().fit_transform(X.astype(np.float32).reshape(len(X), -1) / 255.0).reshape(X.shape)
    X = np.array([resize(img, target_size, anti_aliasing=True) for img in X])[..., None]
    X_small, y_small = resample(X, y, n_samples=int(resample_fraction * len(X)), random_state=random_state)
    n_tr, n_v  = int(len(X_small)*train_size), int(len(X_small)*val_size)
    perm = np.random.RandomState(random_state).permutation(len(X_small))
    tr, v, te = perm[:n_tr], perm[n_tr:n_tr+n_v], perm[n_tr+n_v:]
    x_train, x_val, x_test = map(jnp.array, (X_small[tr], X_small[v], X_small[te]))
    y_train, y_val, y_test = jax.nn.one_hot(y_small[tr], 10), jax.nn.one_hot(y_small[v], 10), jax.nn.one_hot(y_small[te], 10)
    return x_train, x_val, x_test, y_train, y_val, y_test, y_small[tr], y_small[v], y_small[te]

def initialise_edges(adj, key, n_inputs):
    key, sub = random.split(key); Lmat = adj * random.choice(sub, jnp.arange(3., 8., ds), adj.shape)
    key, sk1, sk2 = random.split(key, 3); Wmat = adj * random.uniform(sk1, adj.shape) * ((random.uniform(sk2, adj.shape) > 0.0) * 2 - 1)
    src, tgt = jnp.nonzero(adj)
    L_e = Lmat[src, tgt].at[jnp.where(src < n_inputs)[0]].set(3.0)
    return L_e, Wmat[src, tgt], key

# ==============================================================
# 1. SNN Core Logic (modified)
# ==============================================================
@partial(jit, static_argnames=("dropout_rate", "n_neurons", "n_hidden", "n_outputs", "t_max", "vmax", "thresh", "tau", "dt"))
def rsnn_inference(W_e, L_e, input_currents, rng_key, dropout_rate, src, tgt, n_neurons, n_hidden, n_outputs, t_max, vmax, thresh, tau, dt):
    def _step(carry, key):
        S, V, Vm, acc, phase = carry
        arrived = jnp.isclose(S, L_e, atol=1e-5, rtol=1e-8)
        # I_syn = jax.vmap(lambda ev: jnp.zeros(n_neurons).at[tgt].add(ev))(V * arrived * W_e)
        # is_inject = (phase == 2)
        # I_syn = I_syn.at[:, :n_hidden].add(jnp.where(is_inject, input_currents, 0))
        spikes     = (V * arrived) * W_e                    # (B, E)
        I_syn      = jnp.zeros((B, n_neurons), spikes.dtype)
        I_syn      = I_syn.at[:, tgt].add(spikes)           # scatter-add

        is_inject = (phase == 2)
        I_syn = I_syn.at[:, :n_hidden].add(jnp.where(is_inject, input_currents, 0.0))

        Vm += (-Vm + I_syn) * (dt / tau)
        V_exc = jnp.maximum(0., Vm - thresh)
        fired = (V_exc > 0).at[:, -n_outputs:].set(False)
        def apply_dropout(operands):
            fired_in, V_exc_in, key_in = operands
            mask = random.bernoulli(key_in, 1.0 - dropout_rate, fired_in.shape)
            return fired_in & mask, V_exc_in * mask.astype(jnp.float32) / (1.0 - dropout_rate)
        def no_dropout(operands): return operands[0], operands[1]
        fired, V_exc = lax.cond(dropout_rate > 0.0, apply_dropout, no_dropout, (fired, V_exc, key))
        idle = (S == 0); newS = fired[:, src] & idle
        S, V = S * ~arrived, V * ~arrived
        acc += Vm[:, -n_outputs:]
        Vm -= Vm * fired + 0.2 * fired
        S += (S > 0) * dt * vmax + newS * dt * vmax
        V += newS * V_exc[:, src]
        new_phase = (phase + 1) % 3
        return (S, V, Vm, acc, new_phase), None
    B, E = input_currents.shape[0], len(src)
    carry0 = (
        jnp.zeros((B, E)),             # S
        jnp.zeros((B, E)),             # V
        jnp.zeros((B, n_neurons)),     # Vm
        jnp.zeros((B, n_outputs)),     # acc
        0                              # phase
    )
    keys = random.split(rng_key, t_max)
        # ---------- new: rematerialised scan ----------
    carry_final, _ = lax.scan(
        checkpoint(_step, prevent_cse=False),
        carry0,
        keys,
    )
    _, _, _, acc_final, _ = carry_final
    return acc_final / t_max


# ==============================================================
# 2. Flax NNX Modules (modified)
# ==============================================================
class SmallWorldSNN(nnx.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, k_nearest, p_rewire, p_connect, dropout_rate, *, rngs):
        self.n_inputs, self.n_hidden, self.n_outputs, self.dropout_rate = n_inputs, n_hidden, n_outputs, dropout_rate
        self.n_neurons = n_hidden + n_outputs
        self.t_max, self.vmax, self.thresh, self.tau, self.dt = t_max, vmax, thresh, tau, dt
        key = rngs.params()
        adj, _, key = create_directed_small_world(self.n_inputs + self.n_neurons, self.n_inputs, self.n_outputs, k_nearest, p_rewire, p_connect, key)
        src, tgt = jnp.nonzero(adj)
        L_e_init, W_e_init, _ = initialise_edges(adj, key, self.n_inputs)
        input_mask = src < self.n_inputs
        rec_mask = ~input_mask
        input_src = src[input_mask]
        input_tgt = tgt[input_mask] - self.n_inputs
        rec_src = src[rec_mask] - self.n_inputs
        rec_tgt = tgt[rec_mask] - self.n_inputs
        self.src, self.tgt = rec_src, rec_tgt
        self.L_e = L_e_init[rec_mask]
        self.W_e = nnx.Param(W_e_init[rec_mask])
        self.input_W = nnx.Param(jnp.zeros((self.n_inputs, self.n_hidden)))
        self.input_W.value = self.input_W.value.at[input_src, input_tgt].set(W_e_init[input_mask])
    def __call__(self, x, *, key, training):
        drop = self.dropout_rate if training else 0.0
        input_currents = x.reshape(x.shape[0], -1) @ self.input_W.value
        return rsnn_inference(self.W_e.value, self.L_e, input_currents, key, drop, self.src, self.tgt, self.n_neurons, self.n_hidden, self.n_outputs, self.t_max, self.vmax, self.thresh, self.tau, self.dt)

class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=16, rngs=rngs)
        snn_input_size = 7 * 7 * 16 # 784
        self.snn = SmallWorldSNN(snn_input_size, 256, 10, 16, 0.4, 0.1, 0.2, rngs=rngs)
    def __call__(self, x, *, key: jax.Array, training: bool):
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, (2, 2), (2, 2))
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not training)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, (2, 2), (2, 2))
        return self.snn(x, key=key, training=training)

# ==============================================================
# 3. Training and Evaluation Steps (unchanged)
# ==============================================================
@jit
def train_step(graphdef, params, batch_stats, static_snn, opt_state, x, y, key):
    def loss_fn(params_to_grad):
        model = nnx.merge(graphdef, params_to_grad, batch_stats, static_snn)
        logits = model(x, key=key, training=True)
        loss = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits, axis=1), axis=1))

        # FINAL FIX: The split inside the JIT'd function must also be exhaustive and unpack correctly.
        # nnx.split returns N+1 items for N filters. Here, 3 filters -> 4 items.
        _, _, updated_batch_stats, _ = nnx.split(model, nnx.Param, nnx.BatchStat, ...)
        return loss, updated_batch_stats

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_batch_stats, new_opt_state, loss

@jit
def eval_step(graphdef, params, batch_stats, static_snn, x, y, key):
    model = nnx.merge(graphdef, params, batch_stats, static_snn)
    logits = model(x, key=key, training=False)
    loss = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits, axis=1), axis=1))
    preds = jnp.argmax(logits, axis=1)
    return loss, preds

# ==============================================================
# 4. Main Training Execution (unchanged)
# ==============================================================
(x_train, x_val, x_test, y_train, y_val, y_test, y_train_lbl, y_val_lbl, y_test_lbl) = load_mnist_data(
    train_size=0.8, val_size=0.1, resample_fraction=1.0, target_size=(28,28), random_state=42)

rngs = nnx.Rngs(0)
model = CNN(rngs=rngs)
optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(0.005))

graphdef, params, batch_stats, static_snn = nnx.split(model, nnx.Param, nnx.BatchStat, ...)
opt_state = optimizer.init(params)

batch_size, epochs = 32, 10
rng = random.PRNGKey(42)
print("Starting training with smaller SNN and BatchNorm...")
t0 = time.time()
for ep in range(epochs):
    rng, perm_key = random.split(rng)
    perm = jax.random.permutation(perm_key, len(x_train))
    x_tr, y_tr = x_train[perm], y_train[perm]
    train_loss = 0.0
    for i in range(0, len(x_train), batch_size):
        xb, yb = x_tr[i:i+batch_size], y_tr[i:i+batch_size]
        rng, step_key = random.split(rng)
        params, batch_stats, opt_state, loss = train_step(graphdef, params, batch_stats, static_snn, opt_state, xb, yb, step_key)
        train_loss += float(loss)
    train_loss /= (len(x_train) // batch_size)

    val_loss = 0.0
    all_preds = []
    for j in range(0, len(x_val), batch_size):
        xb, yb = x_val[j:j+batch_size], y_val[j:j+batch_size]
        rng, step_key = random.split(rng)
        loss, preds = eval_step(graphdef, params, batch_stats, static_snn, xb, yb, step_key)
        val_loss += float(loss)
        all_preds.append(preds)
    val_loss /= (len(x_val) // batch_size)
    all_preds = jnp.concatenate(all_preds)
    val_acc = (all_preds == y_val_lbl[:len(all_preds)]).mean() * 100.0
    print(f"Epoch {ep+1:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.2f} %")

print("Training finished in", round(time.time() - t0, 1), "s")

test_preds = []
for i in range(0, len(x_test), batch_size):
    xb, yb = x_test[i:i+batch_size], y_test[i:i+batch_size]
    rng, step_key = random.split(rng)
    _, preds = eval_step(graphdef, params, batch_stats, static_snn, xb, yb, step_key)
    test_preds.append(preds)
test_preds = jnp.concatenate(test_preds)
acc = (test_preds == y_test_lbl[:len(test_preds)]).mean() * 100
print(f"Final test accuracy: {acc:.2f} %")