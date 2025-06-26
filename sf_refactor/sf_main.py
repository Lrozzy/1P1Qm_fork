import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, BSgate, CXgate
import tensorflow as tf
import numpy as np
import h5py
import os, random
import argparse
from helpers.plotting import * 
from circuits import symbolic_circuit
from helpers.utils import load_data, get_loss_fn
from sklearn.metrics import roc_auc_score

# ----------  hyper-params ----------
dim_cutoff      = 10 # fock cutoff dim
wires           = 4 # number of particles per jet (1 wire per particle) DO NOT GO ABOVE 4 (memory blows up)
layers          = 1
steps           = 50
learning_rate   = 0.05
loss_fn         = "bce" # loss function: "bce" or "mse"
train_jets      = 1000
val_jets        = 400
test_jets       = 1000 # Inference is not expensive!

# Paths to data files
data_dir        = "/home/hep/lr1424/1P1Qm_fork/flat_train/TTBar+ZJets_flat.h5"
val_dir         = "/home/hep/lr1424/1P1Qm_fork/flat_val/TTBar+ZJets_flat.h5"
test_dir        = "/home/hep/lr1424/1P1Qm_fork/flat_test/TTBar+ZJets_flat.h5"
save_dir        = "/home/hep/lr1424/1P1Qm_fork/sf_refactor/saved_models_sf"

# Debugging
cli_test = False

parser = argparse.ArgumentParser(description="Run Strawberry Fields SF Simple Model")
parser.add_argument('-name', type=str, help='Name for this run (used for saving models/plots)')
parser.add_argument('--dim_cutoff', type=int, default=dim_cutoff, help='Fock cutoff dimension')
parser.add_argument('--wires', type=int, default=wires, help='Number of wires')
parser.add_argument('--layers', type=int, default=layers, help='Number of layers')
parser.add_argument('--steps', type=int, default=steps, help='Number of training steps')
parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
parser.add_argument('--loss_fn', type=str, default=loss_fn, help='Loss function: "bce" or "mse"')
parser.add_argument('--train_jets', type=int, default=train_jets, help='Number of training jets')
parser.add_argument('--val_jets', type=int, default=val_jets, help='Number of validation jets')
parser.add_argument('--test_jets', type=int, default=test_jets, help='Number of test jets')
parser.add_argument('--data_dir', type=str, default=data_dir, help='Training data file')
parser.add_argument('--val_dir', type=str, default=val_dir, help='Validation data file')
parser.add_argument('--test_dir', type=str, default=test_dir, help='Test data file')
parser.add_argument('--save_dir', type=str, default=save_dir, help='Directory to save models/plots')
parser.add_argument('--cli_test', action='store_true', help='Run in CLI test mode (no saving, no plotting)')
args = parser.parse_args()

# Override hyper-params if specified in args
dim_cutoff    = args.dim_cutoff if args.dim_cutoff else dim_cutoff
wires         = args.wires if args.wires else wires
layers        = args.layers if args.layers else layers
steps         = args.steps if args.steps else steps
learning_rate = args.learning_rate if args.learning_rate else learning_rate
loss_fn       = args.loss_fn if args.loss_fn else loss_fn
train_jets    = args.train_jets if args.train_jets else train_jets
val_jets      = args.val_jets if args.val_jets else val_jets
test_jets     = args.test_jets if args.test_jets else test_jets
data_dir      = args.data_dir if args.data_dir else data_dir
val_dir       = args.val_dir if args.val_dir else val_dir
test_dir      = args.test_dir if args.test_dir else test_dir
save_dir      = args.save_dir if args.save_dir else save_dir
cli_test      = args.cli_test if args.cli_test else cli_test

# Run name
if args.name:
    # If the provided run_name already exists, append _i to make it unique
    base_run_name = args.name
    run_name = base_run_name
    i = 1
    while os.path.exists(f'{save_dir}/{run_name}'):
        run_name = f"{base_run_name}_{i}"
        i += 1
else:
    # Auto-generate a run name like 100_jets_BCE, 20_jets_MSE, etc.
    base_name = f"{train_jets}_jets_{loss_fn}"
    run_name = base_name
    i = 1
    while os.path.exists(f'{save_dir}/{run_name}'):
        run_name = f"{base_name}_{i}"
        i += 1

# Save hyperparameters to a file
if cli_test == False:
    os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
    hyperparams = {
        "dim_cutoff": dim_cutoff,
        "wires": wires,
        "layers": layers,
        "steps": steps,
        "learning_rate": learning_rate,
        "loss_fn": loss_fn,
        "train_jets": train_jets,
        "val_jets": val_jets,
        "test_jets": test_jets,
        "data_dir": data_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "save_dir": save_dir,
        "run_name": run_name,
    }
    hyperparams_path = os.path.join(save_dir, run_name, "hyperparams.txt")
    with open(hyperparams_path, "w") as f:
        for k, v in hyperparams.items():
            f.write(f"{k}: {v}\n")
# -----------------------------------
# Print hyperparams
print("HYPERPARAMETERS:")
print(f"Dimension cutoff: {dim_cutoff}", flush=True)
print(f"Wires: {wires}", flush=True)
print(f"Layers: {layers}", flush=True)
print(f"Steps: {steps}", flush=True)
print(f"Learning rate: {learning_rate}", flush=True)
print(f"Loss function: {loss_fn}", flush=True)
print(f"Train jets: {train_jets}", flush=True)
print(f"Validation jets: {val_jets}", flush=True)
print(f"Test jets: {test_jets}", flush=True)
print(f"Run name: {run_name}", flush=True)
print("------------------------------------", flush=True)

# ----------  load datasets ---------- 
jets, labels = load_data(data_dir, max_jets=train_jets, wires=wires)
jets_val, labels_val = load_data(val_dir, max_jets=val_jets, wires=wires)
jets_test, labels_test = load_data(test_dir, max_jets=test_jets, wires=wires)

# -------- symbolic circuit ----------
# print("Starting symbolic circuit construction...", flush=True)
prog = sf.Program(wires)
s_scale = prog.params("s_scale")
disp_mag  = [prog.params(f"disp_mag{w}") for w in range(wires)] # disp_mag = Displacement Magnitude
disp_phase  = [prog.params(f"disp_phase{w}") for w in range(wires)] # disp_phase = Displacement Phase
squeeze_mag  = [prog.params(f"squeeze_mag{w}") for w in range(wires)] # squeeze_mag = Squeezing Magnitude
squeeze_phase  = [prog.params(f"squeeze_phase{w}") for w in range(wires)] # squeeze_phase = Squeezing Phase
eta = [prog.params(f"eta{w}") for w in range(wires)]
phi = [prog.params(f"phi{w}") for w in range(wires)]
pt  = [prog.params(f"pt{w}")  for w in range(wires)]

weights = {
    's_scale': s_scale,
    **{f'disp_mag_{w}': disp_mag[w] for w in range(wires)},
    **{f'disp_phase_{w}': disp_phase[w] for w in range(wires)},
    **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(wires)},
    **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(wires)},
    **{f'eta_{w}': eta[w] for w in range(wires)},
    **{f'phi_{w}': phi[w] for w in range(wires)},
    **{f'pt_{w}': pt[w] for w in range(wires)},
}

# -------- Circuit architecture ----------
prog = symbolic_circuit(prog, wires, weights)

# -------- Initialise variables ----------
# print("Initialising variables...", flush=True)
rnd = tf.random_uniform_initializer(-0.1, 0.1)
tf_s_scale = tf.Variable(rnd(()))
tf_disp_mag = [tf.Variable(rnd(())) for _ in range(wires)]
tf_disp_phase = [tf.Variable(rnd(())) for _ in range(wires)]
tf_squeeze_mag = [tf.Variable(rnd(())) for _ in range(wires)]
tf_squeeze_phase = [tf.Variable(rnd(())) for _ in range(wires)]

# -------- Feature scaling ----------
# Define assumed limits for features
assumed_limits = {
    'pt':  [1e-4, 3000.0],
    'eta': [-0.8, 0.8],
    'phi': [-0.8, 0.8],
}
# Define feature limits for scaling
feature_limits = {
    'pt':  [0.0, 1.0],
    'eta': [-np.pi, np.pi],
    'phi': [-np.pi, np.pi],
}

def scale_feature(value, name):
    a_min, a_max = assumed_limits[name]
    f_min, f_max = feature_limits[name]
    return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min

# Create arguments for the SF program (map symbolic variables to tensors)
def make_args(jet):
    d = {"s_scale": tf_s_scale}
    for w in range(wires):
        d[f"disp_mag{w}"] = tf_disp_mag[w]
        d[f"disp_phase{w}"] = tf_disp_phase[w]
        d[f"squeeze_mag{w}"] = tf_squeeze_mag[w]
        d[f"squeeze_phase{w}"] = tf_squeeze_phase[w]
        d[f"eta{w}"] = scale_feature(jet[w, 0], "eta")
        d[f"phi{w}"] = scale_feature(jet[w, 1], "phi")
        d[f"pt{w}"]  = scale_feature(jet[w, 2], "pt")
    return d

loss_fn, logit_to_prob = get_loss_fn(loss_fn)
opt = tf.keras.optimizers.Adam(learning_rate)
# print("Starting Engine...", flush=True)
eng = sf.Engine("tf", backend_options={"cutoff_dim": dim_cutoff})

# -------- training loop ----------
print("Starting training...", flush=True)
for step in range(steps):
    # total_train_loss = 0
    idx   = random.randrange(train_jets)
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

    vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase]
    grads = tape.gradient(loss, vars_)
    opt.apply_gradients(zip(grads, vars_))
    
    # total_train_loss += loss

    # -------- validation step ----------
    # total_val_loss = 0
    idx   = random.randrange(val_jets)
    jet_val   = jets_val[idx]
    label_val = labels_val[idx]

    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        state   = eng.run(prog, args=make_args(jet_val)).state
        photons = tf.stack([state.mean_photon(m) for m in range(3)])
        logit   = tf.reduce_sum(photons)            # scalar

        y_true  = tf.expand_dims(label_val, 0)          # shape (1,)
        y_logit = tf.expand_dims(logit, 0)          # shape (1,)
        val_loss    = loss_fn(y_true, y_logit)          

    val_grads = tape.gradient(val_loss, vars_)
    opt.apply_gradients(zip(val_grads, vars_))
        
    if step % 5 == 0:
        # gnorms = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads]
        # print(f"step {step:4d}  loss={loss.numpy():.4f}  "
        #       f"gnorms={['%.1e'%n for n in gnorms[:6]]}", flush=True)
        # print(f"step {step:4d}  loss={val_loss.numpy():.4f}  ", flush=True)
        print(f"Step {step}/{steps} - Training Loss: {loss:.4f} - Validation Loss: {val_loss:.4f}", flush=True)


# --------- Evaluate and print AUC ---------
def predict_prob(jets_tensor):
    """Return an array of P(signal) for each jet."""
    probs = []
    total = jets_tensor.shape[0]
    for i, jet in enumerate(jets_tensor):
        if eng.run_progs:
            eng.reset()
        state   = eng.run(prog, args=make_args(jet)).state
        photons = tf.stack([state.mean_photon(m) for m in range(3)])
        logit   = tf.reduce_sum(photons)
        probs.append(logit_to_prob(logit).numpy())
        if (i+1) % 50 == 0 or (i+1) == total:
            print(f"  Processed {i+1}/{total} jets", flush=True)
    return np.asarray(probs)

# validation ----------------------------------------------------------
# print("Predicting on validation set...", flush=True)
# prob_val = predict_prob(jets_val)
# auc_val  = roc_auc_score(labels_val.numpy(), prob_val)
# print(f"Validation AUC: {auc_val:.4f}", flush=True)

# test ----------------------------------------------------------------
print("Predicting on test set...", flush=True)
prob_test = predict_prob(jets_test)
auc_test  = roc_auc_score(labels_test.numpy(), prob_test)
# print(f"Test AUC: {auc_test:.4f}", flush=True)

# summary -------------------------------------------------
print("Training completed.", flush=True)
# print(f"Final validation AUC: {auc_val:.4f}")
print(f"Final test AUC: {auc_test:.4f}", flush=True)

# Generate and save plots
if cli_test == False:
    plots_dir = os.path.join(save_dir, run_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
    score_hist_path = os.path.join(plots_dir, 'score_histogram.png')
    plot_roc_curve(labels_test.numpy(), prob_test, roc_plot_path)
    plot_score_histogram(labels_test.numpy(), prob_test, score_hist_path)
    print(f"Plots saved to {plots_dir}")