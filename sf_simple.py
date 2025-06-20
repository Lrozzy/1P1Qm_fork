import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, BSgate, CXgate
import tensorflow as tf
import numpy as np
import h5py
from itertools import combinations
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
import matplotlib.pyplot as plt

# --- Global Variables ---
all_wires = None
auto_wires = None
two_comb_wires = None
index = None
num_layers = None
params_per_wire = None

particleFeatureNames = ['eta', 'phi', 'pt']

def getIndex(feat: str) -> int:
    nameArray = particleFeatureNames
    try:
        idx = nameArray.index(feat)
    except ValueError:
        print(f"Feature {feat} not found in {nameArray}")
        idx = -1
    return idx

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def initialize(wires: int = 4, layers: int = 1, params: int = 3):
    """Initializes global variables for the circuit architecture."""
    global all_wires, auto_wires, two_comb_wires, index, num_layers, params_per_wire
    
    all_wires = list(range(wires))
    auto_wires = all_wires[:wires]
    two_comb_wires = list(combinations(range(wires), 2))
    num_layers = layers
    params_per_wire = params
    index = {'eta': getIndex('eta'), 'phi': getIndex('phi'), 'pt': getIndex('pt')}

def load_data_from_h5(file_path, max_jets=None, num_particles=4):
    """Loads jet constituents and truth labels from an H5 file."""
    with h5py.File(file_path, 'r') as f:
        if max_jets:
            # Jet constituents is a list of shape (num_jets, num_particles, num_features) - (40000, 100, 3)
            jet_constituents = f['jetConstituentsList'][:max_jets]
            # Labels are a 1D array of shape (num_jets,) - (40000,) that contains the truth labels (1 for signal, 0 for background)
            labels = f['truth_labels'][:max_jets]
        else:
            jet_constituents = f['jetConstituentsList'][:]
            labels = f['truth_labels'][:]

    # Truncate particles and convert to TensorFlow tensors
    jet_constituents = tf.convert_to_tensor(jet_constituents[:, :num_particles, :], dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    return jet_constituents, labels

def loss_function(y_true, y_pred, loss_type='BCE'):
    """Calculates the loss based on the specified loss_type."""
    if loss_type == 'BCE':
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(y_true, y_pred)
    elif loss_type == 'MSE':
        mse = tf.keras.losses.MeanSquaredError()
        # For MSE with binary labels, we compare against probabilities.
        # The model outputs logits, so we apply sigmoid.
        y_pred_probs = tf.nn.sigmoid(y_pred)
        return mse(y_true, y_pred_probs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Must be 'BCE' or 'MSE'.")


def sf_circuit_template(wires, layers) -> sf.Program:
    """Defines the CV quantum circuit with symbolic parameters."""
    prog = sf.Program(wires)

    # Create symbolic parameters for all weights and inputs
    s_scale = prog.params('s_scale')
    disp_mag_params = [[prog.params(f'disp_mag_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
    disp_phase_params = [[prog.params(f'disp_phase_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
    squeeze_mag_params = [[prog.params(f'squeeze_mag_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
    squeeze_phase_params = [[prog.params(f'squeeze_phase_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
    
    eta_params = [prog.params(f'eta_{w}') for w in range(wires)]
    phi_params = [prog.params(f'phi_{w}') for w in range(wires)]
    pt_params = [prog.params(f'pt_{w}') for w in range(wires)]

    with prog.context as q:
        # Use sf.math for operations on symbolic parameters
        scale_factor = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01

        # Encoding layer
        for w_idx in range(wires):
            Sgate(eta_params[w_idx], pt_params[w_idx] * phi_params[w_idx] / 2.0) | q[w_idx]
            Dgate(scale_factor * pt_params[w_idx], eta_params[w_idx]) | q[w_idx]

        # Trainable layers
        all_wires_list = list(range(wires))
        two_comb_wires = list(combinations(all_wires_list, 2))
        
        for L_idx in range(layers):
            for pair in two_comb_wires:
                CXgate(1.0) | (q[pair[0]], q[pair[1]])
            
            for i in range(wires):
                idx1 = all_wires_list[i]
                idx2 = all_wires_list[(i + 1) % wires]
                BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
            
            for w_idx in range(wires):
                Sgate(squeeze_mag_params[L_idx][w_idx], squeeze_phase_params[L_idx][w_idx]) | q[w_idx]
                Dgate(disp_mag_params[L_idx][w_idx], disp_phase_params[L_idx][w_idx]) | q[w_idx]
            
    return prog

def plot_roc_curve(labels, scores, save_path):
    """Calculates and plots the ROC curve, saving it to a file."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='cornflowerblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: t -> bqq vs q/g jets')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_score_histogram(labels, scores, save_path):
    """Plots and saves a histogram of classifier scores for signal vs background."""
    scores_signal = scores[labels == 1]
    scores_background = scores[labels == 0]
    
    plt.figure()
    plt.hist(scores_background, bins=40, range=(0, 2), color='cornflowerblue', alpha=0.7, label='q/g jets')
    plt.hist(scores_signal, bins=40, range=(0, 2), histtype='step', color='darkorange', lw=2, label='t -> bqq')
    plt.xlabel('Classifier Score')
    plt.ylabel('No. of events')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()
    
class SimpleSFModel(tf.keras.Model):
    def __init__(self, wires=4, layers=1, cutoff_dim=10, output_wires=3, rand_init=True):
        super(SimpleSFModel, self).__init__(dynamic=True)  
        self.wires = wires
        self.num_layers = layers 
        self.cutoff = cutoff_dim
        self.output_wires = output_wires

        # Create the symbolic circuit
        self.prog = sf_circuit_template(wires, layers)

        # Define structured, trainable weights as tf.Variables
        w_init = tf.random_uniform_initializer(-1, 1) if rand_init else tf.ones_initializer()
        self.s_scale = tf.Variable(initial_value=w_init(shape=[], dtype=tf.float32), name='s_scale')
        self.disp_mag_params = [[tf.Variable(initial_value=w_init(shape=[], dtype=tf.float32), name=f'disp_mag_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
        self.disp_phase_params = [[tf.Variable(initial_value=w_init(shape=[], dtype=tf.float32), name=f'disp_phase_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
        self.squeeze_mag_params = [[tf.Variable(initial_value=w_init(shape=[], dtype=tf.float32), name=f'squeeze_mag_L{L}_w{w}') for w in range(wires)] for L in range(layers)]
        self.squeeze_phase_params = [[tf.Variable(initial_value=w_init(shape=[], dtype=tf.float32), name=f'squeeze_phase_L{L}_w{w}') for w in range(wires)] for L in range(layers)]

        # Classical post-processing layer to get a single score
        self.classical_layer = tf.keras.layers.Dense(1, activation=None)

    def _run_single(self, jet):
        # jet shape (wires, 3) : [eta, phi, pt] per wire
        sf_args = {'s_scale': self.s_scale}
        # bind all trainable gate params
        for L in range(self.num_layers):
            for w in range(self.wires):
                sf_args[f'disp_mag_L{L}_w{w}']    = self.disp_mag_params[L][w]
                sf_args[f'disp_phase_L{L}_w{w}']  = self.disp_phase_params[L][w]
                sf_args[f'squeeze_mag_L{L}_w{w}'] = self.squeeze_mag_params[L][w]
                sf_args[f'squeeze_phase_L{L}_w{w}'] = self.squeeze_phase_params[L][w]

        # bind the jet’s features
        eta, phi, pt = jet[:, 0], jet[:, 1], jet[:, 2]
        for w in range(self.wires):
            sf_args[f'eta_{w}'] = eta[w]
            sf_args[f'phi_{w}'] = phi[w]
            sf_args[f'pt_{w}']  = pt[w]

        # run a fresh engine 
        eng   = sf.Engine("tf", backend_options={"cutoff_dim": self.cutoff})
        state = eng.run(self.prog, args=sf_args).state
        photons = tf.stack([state.mean_photon(m) for m in range(self.output_wires)])
        return self.classical_layer(tf.expand_dims(photons, 0))[0, 0]  # scalar

    def call(self, inputs):
        logits = [self._run_single(inputs[i]) for i in range(tf.shape(inputs)[0])]
        return tf.stack(logits, axis=0)  # (batch,)



if __name__ == '__main__':
    # --- Configuration ---
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Run Strawberry Fields SF Simple Model")
    parser.add_argument('-name', type=str, help='Name for this run (used for saving models/plots)')
    args = parser.parse_args()

    if args.name:
        # If the provided run_name already exists, append _i to make it unique
        base_run_name = args.name
        run_name = base_run_name
        i = 1
        while os.path.exists(f'./saved_models_sf/{run_name}'):
            run_name = f"{base_run_name}_{i}"
            i += 1
    else:
        # Auto-generate a run name like base_1, base_2, etc.
        base_name = "base"
        i = 1
        while os.path.exists(f'./saved_models_sf/{base_name}_{i}'):
            i += 1
        run_name = f"{base_name}_{i}"

    NUM_WIRES = 4
    NUM_LAYERS = 1
    CUTOFF_DIM = 10 # Cutoff dimension for the CV quantum circuit
    MAX_JETS_TO_LOAD_TRAIN = 2000 # Out of 40000 jets
    MAX_JETS_TO_LOAD_VAL = 400 # Out of 10000 jets
    MAX_JETS_TO_LOAD_TEST = 400 # Out of 20000 jets
    EPOCHS = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 1 # to start
    sanity_check = False # If True, runs a sanity check to prove gradients are flowing
    rand_init = True # Initalises weights with random values. Only set to False for testing purposes, otherwise True
    loss_type = 'MSE' # 'MSE' or 'BCE' for Binary Cross-Entropy
    save_dir = './saved_models_sf'
    data_dir = '/home/hep/lr1424/1P1Qm_fork/'
    
    # --- Initialization ---
    model = SimpleSFModel(wires=NUM_WIRES, layers=NUM_LAYERS, cutoff_dim=CUTOFF_DIM, output_wires=3, rand_init=rand_init)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Create Save Directories ---
    plots_dir = os.path.join(save_dir, run_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- Load Data ---
    print("--- Loading Data ---")
    train_data_path = os.path.join(data_dir, 'flat_train', 'TTBar+ZJets_flat.h5')
    val_data_path = os.path.join(data_dir, 'flat_val', 'TTBar+ZJets_flat.h5')
    test_data_path = os.path.join(data_dir, 'flat_test', 'TTBar+ZJets_flat.h5')

    train_jets, train_labels = load_data_from_h5(train_data_path, max_jets=MAX_JETS_TO_LOAD_TRAIN, num_particles=NUM_WIRES)
    val_jets, val_labels = load_data_from_h5(val_data_path, max_jets=MAX_JETS_TO_LOAD_VAL, num_particles=NUM_WIRES)
    test_jets, test_labels = load_data_from_h5(test_data_path, max_jets=MAX_JETS_TO_LOAD_TEST, num_particles=NUM_WIRES)

    print(f"Loaded train jets with shape: {train_jets.shape}, labels shape: {train_labels.shape}")
    print(f"Loaded validation jets with shape: {val_jets.shape}, labels shape: {val_labels.shape}")
    print(f"Loaded test jets with shape: {test_jets.shape}, labels shape: {test_labels.shape}")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_jets, train_labels)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_jets, val_labels)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_jets, test_labels)).batch(BATCH_SIZE)

    # Print all configuration parameters
    print(f"Run Name: {run_name}")
    print(f"Number of wires: {NUM_WIRES}, Number of layers: {NUM_LAYERS}, Cutoff dimension: {CUTOFF_DIM}")
    print(f"Max jets to load (train): {MAX_JETS_TO_LOAD_TRAIN}, (val): {MAX_JETS_TO_LOAD_VAL}, (test): {MAX_JETS_TO_LOAD_TEST}")
    print(f"Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"Loss Type: {loss_type}")

    # --- Sanity Check: Prove Gradients are Flowing ---
    if sanity_check:
        print("\n--- Running Sanity Check ---")
        sanity_batch_jets, sanity_batch_labels = next(iter(train_dataset))
        with tf.GradientTape() as tape:
            print(f"Initial scale weight: {model.s_scale.numpy():.4f}")
            # Call the model once to build the layers (including the classical one)
            sanity_scores = model(sanity_batch_jets)
            # Now that the layer is built, we can access its weights
            print(f"Initial first dense weight: {model.classical_layer.weights[0][0].numpy()[0]:.4f}")
            sanity_loss = loss_function(sanity_batch_labels, sanity_scores, loss_type=loss_type)
            print(f"Sanity check loss: {sanity_loss.numpy():.4f}")
        
        sanity_gradients = tape.gradient(sanity_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(sanity_gradients, model.trainable_variables))
        print("Gradients applied.")
        print(f"Weight after 1 step: {model.s_scale.numpy():.4f}")
        print(f"Dense weight after 1 step: {model.classical_layer.weights[0][0].numpy()[0]:.4f}")

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        # --- Training Step ---
        total_train_loss = 0
        num_train_batches = 0
        for batch_jets, batch_labels in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            with tf.GradientTape() as tape:
                # The model is called directly on the batch of jets
                batch_scores = model(batch_jets)
                loss = loss_function(batch_labels, batch_scores, loss_type=loss_type)

            # Gradients are calculated on the model's trainable variables
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            total_train_loss += loss
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # --- Validation Step ---
        total_val_loss = 0
        num_val_batches = 0
        for batch_jets_val, batch_labels_val in tqdm(val_dataset, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            val_scores = model(batch_jets_val)
            val_loss = loss_function(batch_labels_val, val_scores, loss_type=loss_type)
            total_val_loss += val_loss
            num_val_batches += 1
            
        avg_val_loss = total_val_loss / num_val_batches
        
        print(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
        # Print a few weights to see if they change
        # print(f"  -> Scale weight: {model.s_scale.numpy():.4f}, First dense weight: {model.classical_layer.weights[0][0].numpy()[0]:.4f}")

    print("\n--- Finished Training ---")

    # --- Testing Step ---
    print("--- Evaluating on Test Set ---")
    all_test_labels = []
    all_test_scores = []
    for batch_jets_test, batch_labels_test in tqdm(test_dataset, desc="Testing"):
        test_scores = model(batch_jets_test)
        all_test_labels.append(batch_labels_test.numpy())
        all_test_scores.append(test_scores.numpy())
        
    # --- Post-Training Evaluation ---
    all_test_labels = np.concatenate(all_test_labels)
    all_test_scores = np.concatenate(all_test_scores)

    # Use sigmoid on scores for AUC calculation as BCE loss was with logits
    final_scores_for_auc = tf.nn.sigmoid(all_test_scores).numpy()

    auc_score = roc_auc_score(all_test_labels, final_scores_for_auc)
    print(f"Test AUC Score: {auc_score:.4f}")

    # Generate and save plots
    plots_dir = os.path.join(save_dir, run_name, 'plots')
    roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
    score_hist_path = os.path.join(plots_dir, 'score_histogram.png')
    plot_roc_curve(all_test_labels, final_scores_for_auc, roc_plot_path)
    plot_score_histogram(all_test_labels, final_scores_for_auc, score_hist_path)
    print(f"Plots saved to {plots_dir}")


