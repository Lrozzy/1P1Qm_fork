# pylint: disable=maybe-no-member
from typing import Optional, Callable, Union, List, Dict, Tuple, Any
import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, CXgate, MeasureMeanPhoton, BSgate
import tensorflow as tf
import numpy as np # For pi, and initial array conversions if necessary
from helpers.utils import getIndex
from itertools import combinations
import time
import pathlib
import helpers.utils as ut
import os # For path operations in save/load

# Global variable initialization
all_wires: Optional[List[int]] = None
two_comb_wires: Optional[List[Tuple[int, int]]] = None
auto_wires: Optional[List[int]] = None
num_layers: Optional[int] = None
index: Optional[Dict[str, int]] = None
params_per_wire: Optional[int] = None

def sigmoid(x: tf.Tensor) -> tf.Tensor:
    return 1.0 / (1.0 + tf.exp(-tf.cast(x, dtype=tf.float32)))

def initialize(wires: int = 4, layers: int = 1, params: int = 3):
    """
    Initializes the qumode indices and other necessary variables globally.
    """
    global all_wires, auto_wires, two_comb_wires, index, num_layers, params_per_wire
    N_QUMODES = wires
    all_wires = list(range(N_QUMODES))
    params_per_wire = params
    two_comb_wires = list(combinations(list(range(N_QUMODES)), 2))
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
    if backend_name == "tf":
        engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    elif backend_name == "fock":
        engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff_dim})
    else:
        raise ValueError(f"Unsupported Strawberry Fields backend: {backend_name}")
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

    sf_scale = 10.0 * sigmoid(weights[-3]) + 0.01

    with prog.context as q:
        for w_idx in auto_wires:
            current_input_features = inputs_single[w_idx] 

            eta = tf.cast(current_input_features[index['eta']], dtype=tf.float32)
            phi = tf.cast(current_input_features[index['phi']], dtype=tf.float32)
            pt_val = tf.cast(current_input_features[index['pt']], dtype=tf.float32)

            Dgate(sf_scale * pt_val, eta) | q[w_idx]
            Sgate(eta, pt_val * phi / 2.0) | q[w_idx]

        N_auto = len(auto_wires)
        for L_idx in range(num_layers):
            for pair in two_comb_wires:
                CXgate(1.0) | (q[pair[0]], q[pair[1]])
            
            # Add Beamsplitters, similar to the original Pennylane circuit's layer structure
            # Original: qml.Beamsplitter(np.pi/4.,np.pi/2., wires=[w, (w+1)%N])
            # This loop applies BSgate in a ring fashion to the qumodes in auto_wires
            for i in range(N_auto):
                idx1 = auto_wires[i] # Current qumode index
                idx2 = auto_wires[(i + 1) % N_auto] # Next qumode index in a ring
                BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
            
            layer_params_flat_idx_start = L_idx * N_auto * 3
            
            disp_mag_params_layer = weights[layer_params_flat_idx_start : layer_params_flat_idx_start + N_auto]
            disp_phase_params_layer = weights[layer_params_flat_idx_start + N_auto : layer_params_flat_idx_start + 2 * N_auto]
            
            squeeze_mag_params_layer = weights[layer_params_flat_idx_start + 2 * N_auto : layer_params_flat_idx_start + 3 * N_auto]
            fixed_squeeze_phase_layer = tf.constant(np.pi / 4.0, dtype=tf.float32)

            for i, w_idx_loop in enumerate(auto_wires):
                current_disp_mag = disp_mag_params_layer[i]
                current_disp_phase = disp_phase_params_layer[i]
                Dgate(current_disp_mag, current_disp_phase) | q[w_idx_loop]

                current_squeeze_mag = squeeze_mag_params_layer[i]
                Sgate(current_squeeze_mag, fixed_squeeze_phase_layer) | q[w_idx_loop]
                
        num_modes_to_measure = min(len(auto_wires), 3)
        for i in range(num_modes_to_measure):
            w_idx_to_measure = auto_wires[i]
            MeasureMeanPhoton() | q[w_idx_to_measure]
            
    return prog

class QuantumClassifier:
    """
    A class that constructs a Quantum Classifier using Strawberry Fields and TensorFlow.
    """
    def __init__(self, wires:int=4, cutoff_dim: int = 5, sf_backend_name: str = "tf",
                 layers:int=1, params:int=3):
        initialize(wires=wires,layers=layers,params=params)
        self.sf_engine = set_sf_engine(cutoff_dim=cutoff_dim, backend_name=sf_backend_name)
        self.current_weights: Optional[tf.Variable] = None
        self.sf_circuit_template_fn = sf_circuit_template

    def set_circuit(self) -> None:
        """
        Ensures the circuit template is assigned. (Now mostly a placeholder as it's set in __init__)
        """
        if self.sf_circuit_template_fn is None: # Should not happen if __init__ is correct
             self.sf_circuit_template_fn = sf_circuit_template
        print("Strawberry Fields circuit template is set.")

    def fetch_circuit_template(self) -> Callable:
        """
        Retrieves the Strawberry Fields circuit template function.
        """
        if self.sf_circuit_template_fn is None:
            raise RuntimeError("Strawberry Fields circuit template not initialized.")
        return self.sf_circuit_template_fn

    def fetch_engine_backend_name(self) -> str:
        """
        Fetches the backend name of the Strawberry Fields engine.
        """
        return self.sf_engine.backend_name

    def load_weights(self, model_path:str, train:bool=False):
        """
        Loads pre-trained weights. Weights are converted to tf.Variable.
        """
        dictionary = ut.Unpickle(model_path)
        loaded_weights_np = dictionary['weights']
        
        if self.current_weights is None:
            self.current_weights = tf.Variable(initial_value=loaded_weights_np, trainable=train, dtype=tf.float32, name="circuit_weights")
        else:
            # If current_weights exists, update its value and trainable status
            self.current_weights.assign(loaded_weights_np)
            # This is tricky: tf.Variable.trainable cannot be changed after creation directly in all TF versions easily.
            # The safest way if 'trainable' status needs to change is to recreate the variable.
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
        
        # SF 0.23 with TF backend: results.samples for MeasureMeanPhoton is [1, N_modes_measured]
        measured_values = results.samples[0] 
        return tf.cast(measured_values, tf.float32)

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
        num_measured_modes = min(len(auto_wires), 3)
        output_signature = tf.TensorSpec(shape=[num_measured_modes], dtype=tf.float32)

        batch_outputs = tf.map_fn(self.run_circuit_once, batch_data, fn_output_signature=output_signature)
        return batch_outputs

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
                 **kwargs: Any) -> None: # Removed lr_decay, loss_type as they are handled by optimizer and loss_fn_tf
        
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
            # If weights are loaded but marked non-trainable, make them trainable for the trainer
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
        with tf.GradientTape() as tape:
            predictions = self.model.predict_batch(batch_inputs)
            loss = self.loss_function(batch_labels, predictions)

        gradients = tape.gradient(loss, [self.model.current_weights]) 
        self.optimizer.apply_gradients(zip(gradients, [self.model.current_weights]))
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
        Args:
            train_dataset (tf.data.Dataset): Batched training dataset.
            val_dataset (tf.data.Dataset): Batched validation dataset.
            steps_per_epoch_train (Optional[int]): Number of steps per training epoch. If None, iterates full dataset.
            steps_per_epoch_val (Optional[int]): Number of steps per validation epoch. If None, iterates full dataset.
        """
        if self.logger: self.logger.info("Starting training loop...")
        if self.model.current_weights is None:
            raise ValueError("Model weights not initialized before training.")
        if not self.model.current_weights.trainable:
            self.logger.error("CRITICAL: Model weights are not trainable at the start of run_training_loop. Training will not update weights.")
            # This is a critical issue, so we might want to stop or force them trainable if that's intended.
            # Forcing trainable here:
            # self.model.current_weights = tf.Variable(self.model.current_weights.numpy(), trainable=True, name="force_trainable_in_loop")
            # self.logger.info("Forced model weights to be trainable.")
            return # Or raise error

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            total_train_loss = 0.0
            train_steps_taken = 0
            for step, (batch_x, batch_y) in enumerate(train_dataset if steps_per_epoch_train is None else train_dataset.take(steps_per_epoch_train)):
                train_loss = self.train_step(batch_x, batch_y)
                total_train_loss += train_loss.numpy()
                train_steps_taken +=1
                if self.logger and (step % 10 == 0): 
                    self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Step {step+1}/{steps_per_epoch_train or 'all'}, Train Loss: {train_loss.numpy():.4f}")
            
            if train_steps_taken == 0:
                self.logger.warning(f"Epoch {epoch+1}: No training steps were taken. Check train_dataset and steps_per_epoch_train.")
                avg_train_loss = 0.0
            else:
                avg_train_loss = total_train_loss / train_steps_taken
            self.history['train_loss'].append(avg_train_loss)

            total_val_loss = 0.0
            val_steps_taken = 0
            # all_val_labels_list = # For AUC
            # all_val_predictions_list = # For AUC
            for batch_x_val, batch_y_val in enumerate(val_dataset if steps_per_epoch_val is None else val_dataset.take(steps_per_epoch_val)):
                val_loss, predictions = self.validation_step(batch_x_val, batch_y_val) # predictions are tf.Tensor
                total_val_loss += val_loss.numpy()
                val_steps_taken +=1
                # all_val_labels_list.append(batch_y_val.numpy())
                # all_val_predictions_list.append(predictions.numpy())

            if val_steps_taken == 0:
                self.logger.warning(f"Epoch {epoch+1}: No validation steps were taken. Check val_dataset and steps_per_epoch_val.")
                avg_val_loss = float('inf') # Or handle as error
            else:
                avg_val_loss = total_val_loss / val_steps_taken
            self.history['val_loss'].append(avg_val_loss)
            
            # AUC Calculation (example, needs sklearn and adaptation)
            val_auc = -1.0 # Default if not calculated
            # if val_steps_taken > 0 and False: # Disabled for now
            #     try:
            #         from sklearn.metrics import roc_auc_score
            #         val_labels_np = np.concatenate(all_val_labels_list, axis=0)
            #         val_predictions_np = np.concatenate(all_val_predictions_list, axis=0)
            #         # Adjust shapes if necessary for roc_auc_score
            #         if val_labels_np.ndim > 1 and val_labels_np.shape[1] == 1: val_labels_np = val_labels_np.flatten()
            #         if val_predictions_np.ndim > 1 and val_predictions_np.shape[1] == 1: val_predictions_np = val_predictions_np.flatten() # Assuming prediction is a single score for binary
            #         val_auc = roc_auc_score(val_labels_np, val_predictions_np)
            #         self.history['val_auc'].append(val_auc)
            #     except ImportError:
            #         if self.logger: self.logger.warning("scikit-learn not available, AUC not calculated.")
            #     except Exception as e:
            #         if self.logger: self.logger.error(f"Error calculating AUC: {e}")
            # else:
            #     self.history['val_auc'].append(val_auc) # Append default if not calculated


            epoch_duration = time.time() - epoch_start_time
            log_msg = (f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                       f"Val AUC: {val_auc:.4f} (Note: AUC calc needs review), " 
                       f"Duration: {epoch_duration:.2f}s")
            if self.logger: self.logger.info(log_msg)
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                    "val_auc": val_auc, "epoch_duration_s": epoch_duration
                })

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
        save_path = os.path.join(self.checkpoint_dir, file_name)
        
        weights_to_save = {'weights': self.model.current_weights.numpy()}
        ut.Pickle(weights_to_save, file_name, path=self.checkpoint_dir)

        if self.logger:
            self.logger.info(f"Model weights saved to {save_path}")

    def get_current_epoch(self) -> int:
        return self.current_epoch

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.logger: self.logger.info(f"Current epoch set to {self.current_epoch}")

    def fetch_history(self) -> Dict[str, List[float]]:
        return self.history


