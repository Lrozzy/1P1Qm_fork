import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, BSgate, CXgate
import tensorflow as tf, numpy as np, h5py, os, random
from sklearn.metrics import roc_auc_score

# ----------  hyper-params ----------
CUT_OFF     = 10                 # fock cutoff dim
WIRES       = 4
LAYERS      = 1
STEPS       = 50
LR          = 0.01
LOSS_FN     = "mse"               # loss function: "bce" or "mse"
MAX_JETS    = 200                # keep it tiny for the demo
DATA_DIR    = "/home/hep/lr1424/1P1Qm_fork/flat_train/TTBar+ZJets_flat.h5"
VAL_DIR     = "/home/hep/lr1424/1P1Qm_fork/flat_val/TTBar+ZJets_flat.h5"
TEST_DIR    = "/home/hep/lr1424/1P1Qm_fork/flat_test/TTBar+ZJets_flat.h5"
# -----------------------------------

# ----------  load datasets ----------
def load_data(path, max_jets=MAX_JETS):
    with h5py.File(path, "r") as f:
        X = f["jetConstituentsList"][:max_jets, :WIRES, :]     # (N,4,3)
        y = f["truth_labels"][:max_jets].astype(np.float32)     # (N,)
    return tf.convert_to_tensor(X, tf.float32), tf.convert_to_tensor(y, tf.float32)

jets, labels = load_data(DATA_DIR)
jets_val, labels_val = load_data(VAL_DIR)
jets_test, labels_test = load_data(TEST_DIR)

# -------- symbolic circuit ----------
prog = sf.Program(WIRES)
s_scale = prog.params("s_scale")
DM  = [prog.params(f"DM{w}") for w in range(WIRES)]
DP  = [prog.params(f"DP{w}") for w in range(WIRES)]
SM  = [prog.params(f"SM{w}") for w in range(WIRES)]
SP  = [prog.params(f"SP{w}") for w in range(WIRES)]
eta = [prog.params(f"eta{w}") for w in range(WIRES)]
phi = [prog.params(f"phi{w}") for w in range(WIRES)]
pt  = [prog.params(f"pt{w}")  for w in range(WIRES)]

with prog.context as q:
    scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
    for w in range(WIRES):
        Sgate(eta[w], pt[w]*phi[w]/2) | q[w]
        Dgate(scale*pt[w], eta[w])    | q[w]
    for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        CXgate(1.0) | (q[a], q[b])

    all_wires_list = list(range(WIRES))
    for i in range(WIRES):
                idx1 = all_wires_list[i]
                idx2 = all_wires_list[(i + 1) % WIRES]
                BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
    for w in range(WIRES):
        Sgate(SM[w], SP[w]) | q[w]
        Dgate(DM[w], DP[w]) | q[w]

rnd = tf.random_uniform_initializer(-0.1, 0.1)
tf_s_scale = tf.Variable(rnd(()))
tf_DM = [tf.Variable(rnd(())) for _ in range(WIRES)]
tf_DP = [tf.Variable(rnd(())) for _ in range(WIRES)]
tf_SM = [tf.Variable(rnd(())) for _ in range(WIRES)]
tf_SP = [tf.Variable(rnd(())) for _ in range(WIRES)]

assumed_limits = {
    'pt':  [1e-4, 3000.0],
    'eta': [-0.8, 0.8],
    'phi': [-0.8, 0.8],
}
feature_limits = {
    'pt':  [0.0, 1.0],
    'eta': [-np.pi, np.pi],
    'phi': [-np.pi, np.pi],
}

def scale_feature(value, name):
    a_min, a_max = assumed_limits[name]
    f_min, f_max = feature_limits[name]
    return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min

def make_args(jet):
    d = {"s_scale": tf_s_scale}
    for w in range(WIRES):
        d[f"DM{w}"] = tf_DM[w]
        d[f"DP{w}"] = tf_DP[w]
        d[f"SM{w}"] = tf_SM[w]
        d[f"SP{w}"] = tf_SP[w]
        d[f"eta{w}"] = scale_feature(jet[w, 0], "eta")
        d[f"phi{w}"] = scale_feature(jet[w, 1], "phi")
        d[f"pt{w}"]  = scale_feature(jet[w, 2], "pt")
    return d

def get_loss_fn(loss_type='bce'):
    """
    Returns a function loss_fn(y_true, logit) so the training loop
    doesn't need to care whether we are using BCE (logits) or
    MSE (probabilities).
    """
    if loss_type.lower() == "bce":
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # BCE expects logits directly
        def _loss(y_true, logit):
            return bce(y_true, logit)

        # probability to feed to AUC afterwards
        _prob = lambda logit: tf.sigmoid(logit)

    elif loss_type.lower() == "mse":
        mse = tf.keras.losses.MeanSquaredError()

        # MSE should see probabilities in [0,1]
        def _loss(y_true, logit):
            return mse(y_true, tf.sigmoid(logit))

        _prob = lambda logit: tf.sigmoid(logit)

    else:
        raise ValueError(f"Unknown loss type: {loss_type} (use 'bce' or 'mse')")

    return _loss, _prob


loss_fn, logit_to_prob = get_loss_fn(LOSS_FN)
opt = tf.keras.optimizers.Adam(LR)

eng = sf.Engine("tf", backend_options={"cutoff_dim": CUT_OFF})

# -------- training loop ----------
for step in range(STEPS):
    idx   = random.randrange(MAX_JETS)
    jet   = jets[idx]
    label = labels[idx]

    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        state   = eng.run(prog, args=make_args(jet)).state
        photons = tf.stack([state.mean_photon(m) for m in range(3)])
        logit   = tf.reduce_sum(photons)            # scalar

        y_true  = tf.expand_dims(label, 0)          # shape (1,)
        y_logit = tf.expand_dims(logit, 0)          # shape (1,)
        loss    = loss_fn(y_true, y_logit)          

    vars_ = [tf_s_scale, *tf_DM, *tf_DP, *tf_SM, *tf_SP]
    grads = tape.gradient(loss, vars_)
    opt.apply_gradients(zip(grads, vars_))

    if step % 5 == 0:
        gnorms = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads]
        print(f"step {step:4d}  loss={loss.numpy():.4f}  "
              f"gnorms={['%.1e'%n for n in gnorms[:6]]}")

# --------- Evaluate and print AUC ---------
def predict_prob(jets_tensor):
    """Return an array of P(signal) for each jet."""
    probs = []
    for jet in jets_tensor:
        if eng.run_progs:
            eng.reset()
        state   = eng.run(prog, args=make_args(jet)).state
        photons = tf.stack([state.mean_photon(m) for m in range(3)])
        logit   = tf.reduce_sum(photons)
        probs.append(logit_to_prob(logit).numpy())
    return np.asarray(probs)

# validation ----------------------------------------------------------
prob_val = predict_prob(jets_val)
auc_val  = roc_auc_score(labels_val.numpy(), prob_val)
print(f"Validation AUC: {auc_val:.4f}")

# test ----------------------------------------------------------------
prob_test = predict_prob(jets_test)
auc_test  = roc_auc_score(labels_test.numpy(), prob_test)
print(f"Test AUC: {auc_test:.4f}")