# pylint: disable=maybe-no-member
from typing import Optional, Callable, Union, List, Dict, Tuple, Any
import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, CXgate, BSgate
import tensorflow as tf
import numpy as np 
from helpers.utils import getIndex
from itertools import combinations
from tqdm import tqdm
import time
import pathlib
import helpers.utils as ut
import os 
from sklearn.metrics import roc_auc_score

# Global variable initialization
all_wires = None
two_comb_wires = None
auto_wires = None
num_layers = None
index = None
params_per_wire = None

def sigmoid(x: tf.Tensor) -> tf.Tensor:
    return 1.0 / (1.0 + tf.exp(-tf.cast(x, dtype=tf.float32)))

def initialize(wires: int = 4, layers: int = 1, params: int = 3):
    """
    Initializes the qumode indices and other necessary variables globally.
    """
    global all_wires, auto_wires, two_comb_wires, index, num_layers, params_per_wire
    N_QUMODES = wires
    all_wires = [_ for _ in range(N_QUMODES)]
    params_per_wire = params
    two_comb_wires = list(combinations([i for i in range(N_QUMODES)],2))
    auto_wires = all_wires[:N_QUMODES]
    num_layers = layers
    index = {'eta': getIndex('particle', 'eta'), 'phi': getIndex('particle', 'phi'), 'pt': getIndex('particle', 'pt')}

def set_sf_engine(cutoff_dim: int, backend_name: str = "tf") -> sf.Engine:
    """
    Sets the Strawberry Fields engine for simulation.
    Args:
        cutoff_dim (int): Fock space cutoff dimension.
        backend_name (str): Name of the Strawberry Fields backend ("tf", "fock").
    Returns:
        sf.Engine: Initialized Strawberry Fields engine.
    """
    engine = sf.Engine(backend=backend_name, backend_options={"cutoff_dim": cutoff_dim})
    print(f"Strawberry Fields engine initialized with backend: {engine.backend_name}, cutoff_dim: {cutoff_dim}")
    return engine

def print_training_params() -> None:
    """
    Prints out the initialized training parameters for sanity check.
    """
    print("\\n Sanity check: \\n")
    print('all_wires:', all_wires)
    print('auto_wires:', auto_wires)
    print('two_comb_wires:', two_comb_wires)
    print('no. of layers:', num_layers)
    print('index:', index)
    print('trainable parameters per qumode: ', params_per_wire)
    print("\\n ############################################## \\n")
    print("Sleep on it for 3s")
    print("Maybe you want to change something?")
    print("Then press CTRL-C")
    print("\\n ############################################## \\n")
    time.sleep(3)
    print("LETS GOOOOOOOOOOOOO")
    time.sleep(1)

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
            
            # This loop applies BSgate in a ring fashion to the qumodes in auto_wires
            for i in range(N_auto):
                idx1 = auto_wires[i] # Current qumode index
                idx2 = auto_wires[(i + 1) % N_auto] # Next qumode index in a ring
                BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
            
            # Trainable params
            layer_params_flat_idx_start = L_idx * N_auto * 3
            
            # For displacement
            disp_mag_params_layer = weights[layer_params_flat_idx_start : layer_params_flat_idx_start + N_auto]
            disp_phase_params_layer = weights[layer_params_flat_idx_start + N_auto : layer_params_flat_idx_start + 2 * N_auto]
            # For squeezing
            squeeze_mag_params_layer = weights[layer_params_flat_idx_start + 2 * N_auto : layer_params_flat_idx_start + 3 * N_auto]
            squeeze_phase_params_layer = weights[layer_params_flat_idx_start + 3 * N_auto : layer_params_flat_idx_start + 4 * N_auto]

            for i, w_idx_loop in enumerate(auto_wires):
                current_disp_mag = disp_mag_params_layer[i]
                current_disp_phase = disp_phase_params_layer[i]
                Dgate(current_disp_mag, current_disp_phase) | q[w_idx_loop]

                current_squeeze_mag = squeeze_mag_params_layer[i]
                Sgate(current_squeeze_mag, squeeze_phase_params_layer[i]) | q[w_idx_loop]
            
    return prog

class QuantumClassifier:
    """
    A class that constructs a Quantum Classifier using Strawberry Fields and TensorFlow.
    """
    def __init__(self, wires:int, cutoff_dim: int,
                 sf_engine: Optional[sf.Engine] = None,
                 sf_circuit_template_fn: Optional[Callable] = sf_circuit_template):
        """
        Initializes the Quantum Classifier.
        Args:
            wires (int): Number of qumodes in the circuit.
            cutoff_dim (int): Fock space cutoff dimension.
            sf_engine (sf.Engine, optional): Strawberry Fields engine. Defaults to None.
            sf_circuit_template_fn (Callable, optional): Function that defines the circuit. Defaults to sf_circuit_template.
        """
        self.wires = wires
        self.sf_engine = sf_engine if sf_engine is not None else set_sf_engine(cutoff_dim=cutoff_dim)
        self.sf_circuit_template_fn = sf_circuit_template_fn
        self.current_weights: Optional[tf.Variable] = None

    def load_weights(self, model_path:str, train:bool=False):
        """
        Loads pre-trained weights. Weights are converted to tf.Variable.
        """
        dictionary = ut.Unpickle(model_path)
        loaded_weights_np = dictionary['weights']

        if self.current_weights is None:
            self.current_weights = tf.Variable(initial_value=loaded_weights_np, trainable=train, dtype=tf.float32, name="circuit_weights")
        else:
            self.current_weights.assign(loaded_weights_np)
            if self.current_weights.trainable != train:
                 self.current_weights = tf.Variable(initial_value=self.current_weights.numpy(), trainable=train, dtype=tf.float32, name="circuit_weights_recreated")
        
        print(f"Weights loaded from {model_path}. Trainable: {self.current_weights.trainable}")

    def print_weights(self):
        """
        Prints the current weights.
        """
        if self.current_weights is not None:
            print('Current weights: \\n\\n', self.current_weights.numpy())
        else:
            print("Weights not loaded or initialized.")

    def run_circuit_once(self, single_input_instance: tf.Tensor) -> tf.Tensor:
        """
        Runs the SF circuit for a single input instance using the current weights.
        Args:
            single_input_instance (tf.Tensor): A single data instance, shape [num_qumodes, num_features].
        Returns:
            tf.Tensor: The measurement results from the circuit (e.g., mean photon numbers).
        """
        if self.current_weights is None:
            raise ValueError("Weights not initialized. Load or set weights first.")
        if not isinstance(single_input_instance, tf.Tensor):
            single_input_instance = tf.convert_to_tensor(single_input_instance, dtype=tf.float32)

        prog = self.sf_circuit_template_fn(self.current_weights, single_input_instance)
        results = self.sf_engine.run(prog)
        state = results.state                                

        # Grab mean photon numbers; each call returns a scalar tf.Tensor
        # Applying tf.reduce_mean as a workaround for unexpected tensor shapes from the backend.
        nbar = [tf.reduce_mean(state.mean_photon(m)) for m in range(self.wires)]
        return tf.stack(nbar)

    def predict_batch(self, batch_data: tf.Tensor) -> tf.Tensor:
        """
        Runs inference on a batch of data using tf.map_fn.
        Args:
            batch_data (tf.Tensor): Batch of input data, shape [batch_size, num_qumodes, num_features].
        Returns:
            tf.Tensor: Circuit outputs for the batch, shape [batch_size, num_measured_modes].
        """
        if not isinstance(batch_data, tf.Tensor):
            batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        
        # Define the output signature for tf.map_fn
        # num_measured_modes depends on the circuit, here it's min(len(auto_wires), 3)
        if auto_wires is None: # Should be initialized
            raise ValueError("auto_wires not initialized. Call initialize() first.")
        num_measured_modes = self.wires
        output_signature = tf.TensorSpec(shape=[num_measured_modes], dtype=tf.float32)

        circuit_outputs = tf.map_fn(self.run_circuit_once, batch_data, fn_output_signature=output_signature)
        
        # Apply scaling and bias
        mean_photon_number = tf.reduce_mean(circuit_outputs, axis=1)
        scores = sigmoid(mean_photon_number)
        return scores

    def get_predictions(self, data: np.ndarray) -> np.ndarray:
        """
        Gets raw predictions from the model for the given data.
        Args:
            data (np.ndarray): Input data, assumed to be preprocessed to [num_samples, num_qumodes, num_features].
        Returns:
            np.ndarray: Model outputs (e.g., mean photon numbers).
        """
        if self.current_weights is None:
            raise ValueError('Weights not initialized. Load a model first by calling load_weights(model_path)')
        
        data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
        
        # Basic check for input shape if auto_wires is initialized
        if auto_wires is not None and data_tf.shape.rank == 3:
            if data_tf.shape[1] != len(auto_wires):
                 print(f"Warning: Input data's second dimension ({data_tf.shape[1]}) does not match number of auto_wires ({len(auto_wires)}).")
        elif data_tf.shape.rank != 3:
            print(f"Warning: Input data rank is {data_tf.shape.rank}, expected 3 ([batch, qumodes, features]).")


        predictions_tf = self.predict_batch(data_tf)
        return predictions_tf.numpy()

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Returns all trainable variables of the model."""
        t_vars = []
        if self.current_weights is not None and self.current_weights.trainable:
            t_vars.append(self.current_weights)
        return t_vars

class QuantumTrainer:
    """
    A class for training the QuantumClassifier using Strawberry Fields and TensorFlow.
    """
    def __init__(self, model: QuantumClassifier,
                 optimizer_tf: tf.keras.optimizers.Optimizer,
                 loss_fn_tf: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 save: bool = True, epochs: int = 20, patience: int = 2,
                 improv: float = 0.01, wandb_run: Optional[Any] = None,
                 batch_size: int = 32, logger: Optional[Any] = None,
                 init_weights_val: Optional[np.ndarray] = None,
                 save_dir_path: str = "saved_models_sf",
                 **kwargs: Any) -> None:

        self.model = model
        self.optimizer = optimizer_tf
        self.loss_function = loss_fn_tf

        if init_weights_val is not None:
            if self.model.current_weights is None:
                self.model.current_weights = tf.Variable(initial_value=init_weights_val, trainable=True, dtype=tf.float32, name="circuit_weights_trainer_init")
            else:
                self.model.current_weights.assign(init_weights_val)
                if not self.model.current_weights.trainable:
                    self.model.current_weights = tf.Variable(initial_value=self.model.current_weights.numpy(), trainable=True, dtype=tf.float32, name="circuit_weights_trainer_reinit_trainable")
        elif self.model.current_weights is None:
            raise ValueError("QuantumTrainer: model.current_weights are None and no init_weights_val provided. Initialize or load weights into the model first.")
        elif not self.model.current_weights.trainable:
            print("Warning: Model weights were non-trainable. Recreating as trainable for the QuantumTrainer.")
            self.model.current_weights = tf.Variable(initial_value=self.model.current_weights.numpy(), trainable=True, dtype=tf.float32, name="circuit_weights_trainer_make_trainable")

        self.batch_size = batch_size
        self.logger = logger
        self.epochs = epochs
        self.patience = patience
        self.saving = save
        self.current_epoch = 0
        self.improv = improv
        self.is_evictable = False
        self.seed = None
        self.wandb_run = wandb_run
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_auc': []}

        self.save_dir = save_dir_path
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints_sf')
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if self.logger:
            self.logger.info(f"QuantumTrainer initialized. Optimizer: {self.optimizer.get_config()}")
            self.logger.info(f"Model weights trainable: {self.model.current_weights.trainable}")

    @tf.function
    def train_step(self, batch_inputs: tf.Tensor, batch_labels: tf.Tensor) -> tf.Tensor:
        """Performs a single training step."""
        trainable_vars = self.model.get_trainable_variables()
        with tf.GradientTape() as tape:
            predictions = self.model.predict_batch(batch_inputs)
            loss = self.loss_function(batch_labels, predictions)

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss

    @tf.function
    def validation_step(self, batch_inputs: tf.Tensor, batch_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs a single validation step."""
        predictions = self.model.predict_batch(batch_inputs)
        loss = self.loss_function(batch_labels, predictions)
        return loss, predictions

    def run_training_loop(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                          steps_per_epoch_train: Optional[int] = None,
                          steps_per_epoch_val: Optional[int] = None):
        """
        Executes the full training loop.
        """
        if self.logger: self.logger.info("Starting training loop...")
        if self.model.current_weights is None:
            raise ValueError("Model weights not initialized before training.")
        if not self.model.current_weights.trainable:
            self.logger.error("CRITICAL: Model weights are not trainable at the start of run_training_loop.")
            return

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # --- Training Loop ---
            total_train_loss = 0.0
            train_steps_taken = 0
            train_iterator = tqdm(train_dataset.take(steps_per_epoch_train) if steps_per_epoch_train else train_dataset,
                                  desc=f"Epoch {epoch+1}/{self.epochs} [Train]", unit="batch", leave=False)

            for batch_x, batch_y in train_iterator:
                train_loss = self.train_step(batch_x, batch_y)
                loss_val = train_loss.numpy()
                total_train_loss += loss_val
                train_steps_taken += 1
                train_iterator.set_postfix(loss=f"{loss_val:.4f}")

            avg_train_loss = total_train_loss / train_steps_taken if train_steps_taken > 0 else 0.0
            self.history['train_loss'].append(avg_train_loss)

            # --- Validation Loop ---
            total_val_loss = 0.0
            val_steps_taken = 0
            all_val_labels_list = []
            all_val_predictions_list = []
            val_iterator = tqdm(val_dataset.take(steps_per_epoch_val) if steps_per_epoch_val else val_dataset,
                                desc=f"Epoch {epoch+1}/{self.epochs} [Val]", unit="batch", leave=False)

            for batch_x_val, batch_y_val in val_iterator:
                val_loss, predictions = self.validation_step(batch_x_val, batch_y_val)
                loss_val = val_loss.numpy()
                total_val_loss += loss_val
                val_steps_taken += 1
                all_val_labels_list.append(batch_y_val.numpy())
                all_val_predictions_list.append(predictions.numpy())
                val_iterator.set_postfix(loss=f"{loss_val:.4f}")

            avg_val_loss = total_val_loss / val_steps_taken if val_steps_taken > 0 else float('inf')
            self.history['val_loss'].append(avg_val_loss)

            # --- AUC Calculation ---
            val_auc = -1.0
            if val_steps_taken > 0:
                try:
                    val_labels_np = np.concatenate(all_val_labels_list, axis=0)
                    val_predictions_np = np.concatenate(all_val_predictions_list, axis=0)
                    scores_for_auc = val_predictions_np[:, 0] if val_predictions_np.ndim > 1 else val_predictions_np
                    if val_labels_np.ndim > 1 and val_labels_np.shape[1] == 1:
                        val_labels_np = val_labels_np.flatten()
                    val_auc = roc_auc_score(val_labels_np, scores_for_auc)
                except Exception as e:
                    if self.logger: self.logger.error(f"Could not compute AUC: {e}")
            self.history['val_auc'].append(val_auc)

            # --- Epoch End Logging & Checkpointing ---
            epoch_duration = time.time() - epoch_start_time
            log_msg = (f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                       f"Val AUC: {val_auc:.4f}, Duration: {epoch_duration:.2f}s")
            if self.logger: self.logger.info(log_msg)
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                    "val_auc": val_auc, "epoch_duration_s": epoch_duration
                })

            # Early stopping and model saving
            if avg_val_loss < best_val_loss - self.improv:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if self.saving:
                    self.save_model_weights(epoch_identifier=f"epoch_{epoch+1}_best")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.logger: self.logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        if self.saving:
            self.save_model_weights(epoch_identifier="final_model")
        if self.logger: self.logger.info("Training loop finished.")
        return self.history

    def print_params(self, prefix: Optional[str] = None) -> None:
        if prefix: print(prefix)
        if self.model.current_weights is not None:
            print('Model weights (numpy):\\n', self.model.current_weights.numpy())
        else:
            print("Model weights not set.")

    def save_model_weights(self, epoch_identifier: str) -> None:
        """Saves the model weights as a pickle file."""
        if self.model.current_weights is None:
            if self.logger: self.logger.warning("Attempted to save model, but weights are None.")
            return

        file_name = f"model_weights_{epoch_identifier}.pickle"
        weights_to_save = {
            'weights': self.model.current_weights.numpy()
        }
        ut.Pickle(weights_to_save, file_name, path=self.checkpoint_dir)

        if self.logger:
            self.logger.info(f"Model weights saved to {os.path.join(self.checkpoint_dir, file_name)}")

    def get_current_epoch(self) -> int:
        return self.current_epoch

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.logger: self.logger.info(f"Current epoch set to {self.current_epoch}")

    def fetch_history(self) -> Dict[str, List[float]]:
        return self.history


