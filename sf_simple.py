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

def loss_function(y_true, y_pred, type='BCE'):
    """Calculates the loss between the true and predicted values."""
    if type == 'MSE':
        # Expand dims of y_true to allow broadcasting with y_pred's shape [batch, 3]
        if len(y_true.shape) == 1 and len(y_pred.shape) == 2:
            y_true = tf.expand_dims(y_true, 1)
        return tf.reduce_mean(tf.square(y_true - y_pred))
    elif type == 'BCE':
        # For BCE, reduce y_pred to a single score per instance (e.g., sum or mean)
        if len(y_pred.shape) == 2:
            y_pred_score = tf.reduce_sum(y_pred, axis=1)
        else:
            y_pred_score = y_pred
        # Ensure y_true shape matches y_pred_score
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(y_true, y_pred_score)
    else:
        raise ValueError(f"Unsupported loss type: {type}")

def sf_circuit_template(weights: tf.Tensor, inputs: tf.Tensor) -> sf.Program:
    """
    Defines the CV quantum circuit using Strawberry Fields.
    Args:
        weights (tf.Tensor): Trainable parameters.
        inputs (tf.Tensor): Input data. Assumes inputs are correctly shaped for TF.
                            Expected shape for inputs: [num_qumodes, num_features_per_qumode]
                            This function is designed for ONE instance from a batch.
                            Batching (e.g. tf.map_fn over inputs) should be handled by the caller.
    Returns:
        sf.Program: The Strawberry Fields program.
    """
    if all_wires is None or auto_wires is None or index is None or num_layers is None or two_comb_wires is None:
        raise ValueError("Global parameters (all_wires, auto_wires, etc.) not initialized. Call initialize() first.")

    prog = sf.Program(len(all_wires))

    if not isinstance(weights, tf.Tensor):
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    if not isinstance(inputs, tf.Tensor):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    if tf.rank(inputs) != 2: # (num_qumodes, features)
        raise ValueError(f"sf_circuit_template expects single instance inputs with rank 2 ([num_qumodes, num_features]). Got rank {tf.rank(inputs)}. Batch with tf.map_fn.")
    
    inputs_single = inputs # Explicitly state we are working with a single instance

    scale_factor = 10.0 * sigmoid(weights[-3]) + 0.01

    with prog.context as q:
        for w_idx in auto_wires:
            current_input_features = inputs_single[w_idx]

            eta = tf.cast(current_input_features[index['eta']], dtype=tf.float32)
            phi = tf.cast(current_input_features[index['phi']], dtype=tf.float32)
            pt = tf.cast(current_input_features[index['pt']], dtype=tf.float32)

            Dgate(scale_factor * pt, eta) | q[w_idx]
            Sgate(eta, pt * phi / 2.0) | q[w_idx]

        N_auto = len(auto_wires)
        for L_idx in range(num_layers):
            for pair in two_comb_wires:
                CXgate(1.0) | (q[pair[0]], q[pair[1]])
            
            for i in range(N_auto):
                idx1 = auto_wires[i] 
                idx2 = auto_wires[(i + 1) % N_auto] 
                BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
            
            layer_params_flat_idx_start = L_idx * N_auto * 4
            
            disp_mag_params_layer = weights[layer_params_flat_idx_start : layer_params_flat_idx_start + N_auto]
            disp_phase_params_layer = weights[layer_params_flat_idx_start + N_auto : layer_params_flat_idx_start + 2 * N_auto]
            squeeze_mag_params_layer = weights[layer_params_flat_idx_start + 2 * N_auto : layer_params_flat_idx_start + 3 * N_auto]
            squeeze_phase_params_layer = weights[layer_params_flat_idx_start + 3 * N_auto : layer_params_flat_idx_start + 4 * N_auto]

            for i, w_idx_loop in enumerate(auto_wires):
                Dgate(disp_mag_params_layer[i], disp_phase_params_layer[i]) | q[w_idx_loop]
                Sgate(squeeze_mag_params_layer[i], squeeze_phase_params_layer[i]) | q[w_idx_loop]
            
    return prog

class SimpleSFModel:
    def __init__(self, wires=4, layers=1, cutoff_dim=10, backend_name='tf', output_wires=3, rand_init=True):
        self.wires = wires
        self.layers = layers
        self.output_wires = output_wires
        self.sf_engine = sf.Engine(backend=backend_name, backend_options={"cutoff_dim": cutoff_dim})
        self.sf_circuit_template_fn = sf_circuit_template
        
        # Initialize weights as a trainable variable
        num_weights = (layers * wires * 4) + 3
        self.current_weights = tf.Variable(tf.random.uniform([num_weights], -1, 1, dtype=tf.float32)) if rand_init else tf.Variable(tf.ones([num_weights], dtype=tf.float32))
        
        initialize(wires=self.wires, layers=self.layers)

    def run_circuit_once(self, single_input_instance: tf.Tensor) -> tf.Tensor:
        if self.current_weights is None:
            raise ValueError("Weights not initialized. Load or set weights first.")
        if not isinstance(single_input_instance, tf.Tensor):
            single_input_instance = tf.convert_to_tensor(single_input_instance, dtype=tf.float32)

        prog = self.sf_circuit_template_fn(self.current_weights, single_input_instance)
        results = self.sf_engine.run(prog)
        state = results.state                                

        nbar = [tf.reduce_mean(state.mean_photon(m)) for m in range(self.output_wires)]
        return tf.stack(nbar)

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
    CUTOFF_DIM = 5
    MAX_JETS_TO_LOAD_TRAIN = 100 # Out of 40000 jets
    MAX_JETS_TO_LOAD_VAL = 20 # Out of 10000 jets
    MAX_JETS_TO_LOAD_TEST = 20 # Out of 20000 jets
    EPOCHS = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 10
    rand_init = False # Initalises weights with random values. Only set to False for testing purposes, otherwise True
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

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        # --- Training Step ---
        total_train_loss = 0
        num_train_batches = 0
        for batch_jets, batch_labels in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            with tf.GradientTape() as tape:
                batch_outputs = tf.map_fn(lambda x: model.run_circuit_once(x), batch_jets, dtype=tf.float32)
                # Loss is calculated directly on the 3-dim output vs the scalar label (handled by broadcasting)
                loss = loss_function(batch_labels, batch_outputs, type=loss_type)

            # Calculate and apply gradients
            gradients = tape.gradient(loss, [model.current_weights])
            optimizer.apply_gradients(zip(gradients, [model.current_weights]))
            
            total_train_loss += loss
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # --- Validation Step ---
        total_val_loss = 0
        num_val_batches = 0
        for batch_jets_val, batch_labels_val in tqdm(val_dataset, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            batch_outputs_val = tf.map_fn(lambda x: model.run_circuit_once(x), batch_jets_val, dtype=tf.float32)
            val_loss = loss_function(batch_labels_val, batch_outputs_val)
            total_val_loss += val_loss
            num_val_batches += 1
            
        avg_val_loss = total_val_loss / num_val_batches
        
        print(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
        print(f"  -> First 5 weights: {model.current_weights.numpy()[:5]}")

    print("\n--- Finished Training ---")

    # --- Testing Step ---
    print("--- Evaluating on Test Set ---")
    total_test_loss = 0
    num_test_batches = 0
    all_test_labels = []
    all_test_scores = []
    for batch_jets_test, batch_labels_test in tqdm(test_dataset, desc="Testing"):
        batch_outputs_test = tf.map_fn(lambda x: model.run_circuit_once(x), batch_jets_test, dtype=tf.float32)
        test_loss = loss_function(batch_labels_test, batch_outputs_test)
        
        # For evaluation (AUC/ROC), we convert the output to a single score
        scores_test = tf.reduce_sum(batch_outputs_test, axis=1)

        total_test_loss += test_loss
        num_test_batches += 1
        all_test_labels.append(batch_labels_test.numpy())
        all_test_scores.append(scores_test.numpy())
        
    avg_test_loss = total_test_loss / num_test_batches
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    # --- Post-Training Evaluation ---
    all_test_labels = np.concatenate(all_test_labels)
    all_test_scores = np.concatenate(all_test_scores)

    auc_score = roc_auc_score(all_test_labels, all_test_scores)
    print(f"Test AUC Score: {auc_score:.4f}")

    # Generate and save plots
    roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
    score_hist_path = os.path.join(plots_dir, 'score_histogram.png')
    plot_roc_curve(all_test_labels, all_test_scores, roc_plot_path)
    plot_score_histogram(all_test_labels, all_test_scores, score_hist_path)
    print(f"Plots saved to {plots_dir}")

    # --- Run a single instance post-training to see the result ---
    print("--- Running a single instance post-training ---")
    first_instance_input = test_jets[0]
    first_instance_label = test_labels[0]
    output_photons = model.run_circuit_once(first_instance_input)
    final_score = tf.reduce_sum(output_photons)
    print(f"Input shape: {first_instance_input.shape}")
    print(f"True Label: {first_instance_label.numpy()}")
    print(f"Output (mean photons): {output_photons.numpy()}")
    print(f"Final Score: {final_score.numpy():.4f}")
