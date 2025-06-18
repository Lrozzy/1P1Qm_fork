import unittest
import os
import shutil
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock
import sys
sys.path.insert(0, '/home/hep/lr1424/1P1Qm_fork')
from quantum.architectures import QuantumTrainer, QuantumClassifier, initialize, sigmoid
from sklearn.metrics import roc_auc_score

# Use an absolute import as the parent is a namespace package
import helpers.utils as ut

class TestQuantumTrainerSaveWeights(unittest.TestCase):
    """
    Tests for the save_model_weights method of the QuantumTrainer class.
    """

    def setUp(self):
        """
        Set up a temporary directory and a QuantumTrainer instance for each test.
        """
        self.test_dir = "temp_test_save_weights_dir"
        # Ensure the directory is clean before starting
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        # Initialize global parameters required by the architecture
        initialize(wires=4, layers=1) 

        # Create a mock logger to spy on log messages
        self.mock_logger = MagicMock()

        # Instantiate the model and its dependencies
        self.classifier = QuantumClassifier(4, 5)
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = lambda y_true, y_pred: tf.constant(0.0)

        # Define initial weights. Based on initialize(wires=4, layers=1):
        # 4 qumodes * 4 params/qumode/layer * 1 layer + 3 global params = 19
        num_weights = 4 * 4 * 1 + 3
        self.initial_weights = np.random.rand(num_weights).astype(np.float32)

        # Instantiate the trainer
        self.trainer = QuantumTrainer(
            model=self.classifier,
            optimizer_tf=optimizer,
            loss_fn_tf=loss_fn,
            logger=self.mock_logger,
            init_weights_val=self.initial_weights,
            save_dir_path=self.test_dir
        )

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        """
        shutil.rmtree(self.test_dir)

    def test_save_model_weights_creates_file(self):
        """
        Verify that save_model_weights creates a file at the expected location.
        """
        # Arrange
        epoch_id = "epoch_1_test"
        expected_filename = f"model_weights_{epoch_id}.pickle"
        expected_filepath = os.path.join(self.trainer.checkpoint_dir, expected_filename)

        # Act
        self.trainer.save_model_weights(epoch_identifier=epoch_id)

        # Assert
        self.assertTrue(os.path.exists(expected_filepath), "Model weights file was not created.")

    def test_save_model_weights_correct_content(self):
        """
        Verify that the saved file contains the correct weights.
        """
        # Arrange
        epoch_id = "content_check"
        filename = f"model_weights_{epoch_id}.pickle"
        filepath = os.path.join(self.trainer.checkpoint_dir, filename)

        # Act
        self.trainer.save_model_weights(epoch_identifier=epoch_id)

        # Assert
        saved_data = ut.Unpickle(filepath)

        self.assertIsInstance(saved_data, dict, "Saved data should be a dictionary.")
        self.assertIn('weights', saved_data, "The key 'weights' should be in the saved dictionary.")
        self.assertNotIn('scale', saved_data, "The key 'scale' should not be in the saved dictionary.")
        self.assertNotIn('bias', saved_data, "The key 'bias' should not be in the saved dictionary.")
        
        saved_weights = saved_data['weights']
        np.testing.assert_array_equal(saved_weights, self.initial_weights,
                                      "The weights saved to the file do not match the initial weights.")

    def test_save_model_weights_logs_info_message_on_success(self):
        """
        Verify that an info message is logged upon successful saving.
        """
        # Arrange
        epoch_id = "log_check"
        expected_filename = f"model_weights_{epoch_id}.pickle"
        expected_filepath = os.path.join(self.trainer.checkpoint_dir, expected_filename)

        # Reset the mock logger to ignore calls made during initialization
        self.mock_logger.reset_mock()

        # Act
        self.trainer.save_model_weights(epoch_identifier=epoch_id)

        # Assert
        self.mock_logger.info.assert_called_once_with(
            f"Model weights saved to {expected_filepath}"
        )

    def test_save_model_weights_handles_none_weights(self):
        """Verify correct behavior when model weights are None."""
        self.mock_logger.reset_mock()
        self.trainer.model.current_weights = None
        self.trainer.save_model_weights("test_epoch")
        self.mock_logger.warning.assert_called_with("Attempted to save model, but weights are None.")
        self.mock_logger.info.assert_not_called()

class TestQuantumClassifier(unittest.TestCase):
    """
    Tests for the QuantumClassifier class.
    """
    def setUp(self):
        """
        Set up a QuantumClassifier instance for each test.
        """
        self.test_dir = "temp_test_classifier_dir"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        initialize(wires=2, layers=1)
        self.classifier = QuantumClassifier(2, 5)
        self.num_weights = 2 * 4 * 1 + 3 # 11 weights for this config

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        """
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """
        Verify that the QuantumClassifier is initialized correctly.
        """
        self.assertIsNotNone(self.classifier.sf_engine, "SF engine should be initialized.")
        self.assertEqual(self.classifier.sf_engine.backend_name, "tf", "SF engine backend should be 'tf'.")
        self.assertIsNone(self.classifier.current_weights, "Initial weights should be None.")
        self.assertFalse(hasattr(self.classifier, 'scale'), "Classifier should not have a 'scale' attribute.")
        self.assertFalse(hasattr(self.classifier, 'bias'), "Classifier should not have a 'bias' attribute.")

    def test_load_weights(self):
        """
        Test loading weights from a pickle file.
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        weights_dict = {'weights': test_weights}
        
        model_path = os.path.join(self.test_dir, "test_weights.pickle")
        ut.Pickle(weights_dict, "test_weights.pickle", path=self.test_dir)

        # Act
        self.classifier.load_weights(model_path, train=True)

        # Assert
        self.assertIsNotNone(self.classifier.current_weights, "Weights should be loaded.")
        self.assertTrue(self.classifier.current_weights.trainable, "Loaded weights should be trainable.")
        np.testing.assert_array_equal(self.classifier.current_weights.numpy(), test_weights, "Loaded weights do not match saved weights.")

    def test_load_weights_updates_trainable_status(self):
        """
        Test that loading weights can change the trainable status of the weights tensor.
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        weights_dict = {'weights': test_weights}
        model_path = os.path.join(self.test_dir, "trainable_test.pickle")
        ut.Pickle(weights_dict, "trainable_test.pickle", path=self.test_dir)

        # Act & Assert
        # 1. Load as not trainable
        self.classifier.load_weights(model_path, train=False)
        self.assertFalse(self.classifier.current_weights.trainable, "Weights should be non-trainable after loading with train=False.")

        # 2. Load again as trainable
        self.classifier.load_weights(model_path, train=True)
        self.assertTrue(self.classifier.current_weights.trainable, "Weights should be trainable after loading with train=True.")

    def test_get_trainable_variables(self):
        """
        Test that get_trainable_variables returns all trainable parameters correctly.
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        self.classifier.current_weights = tf.Variable(test_weights, trainable=True, name="test_weights")

        # Act
        trainable_vars = self.classifier.get_trainable_variables()

        # Assert
        self.assertEqual(len(trainable_vars), 1, "Should return 1 trainable variable: weights.")
        # Use identity comparison (is) for TensorFlow variables
        self.assertTrue(any(var is self.classifier.current_weights for var in trainable_vars), "Weights should be in trainable variables.")

    def test_get_trainable_variables_without_weights(self):
        """
        Test get_trainable_variables when current_weights is None.
        """
        # Arrange
        self.classifier.current_weights = None

        # Act
        trainable_vars = self.classifier.get_trainable_variables()

        # Assert
        self.assertEqual(len(trainable_vars), 0, "Should return 0 trainable variables when there are no weights.")

    def test_get_trainable_variables_with_non_trainable_weights(self):
        """
        Test get_trainable_variables when current_weights is not trainable.
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        self.classifier.current_weights = tf.Variable(test_weights, trainable=False, name="non_trainable_weights")

        # Act
        trainable_vars = self.classifier.get_trainable_variables()

        # Assert
        self.assertEqual(len(trainable_vars), 0, "Should return 0 trainable variables when weights are non-trainable.")
        self.assertFalse(any(var is self.classifier.current_weights for var in trainable_vars), "Non-trainable weights should not be in trainable variables.")

    def test_run_circuit_once_raises_error_when_no_weights(self):
        """
        Test that run_circuit_once raises ValueError when current_weights is None.
        """
        # Arrange
        self.classifier.current_weights = None
        sample_input = tf.random.normal([2, 3])  # [num_qumodes, num_features]

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.classifier.run_circuit_once(sample_input)
        
        self.assertIn("Weights not initialized. Load or set weights first.", str(context.exception))

    def test_run_circuit_once_returns_tensor(self):
        """
        Test that run_circuit_once returns a TensorFlow tensor with correct shape.
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        self.classifier.current_weights = tf.Variable(test_weights, trainable=True, name="test_weights")
        sample_input = tf.random.normal([2, 3])  # [num_qumodes, num_features]

        # Act
        result = self.classifier.run_circuit_once(sample_input)

        # Assert
        self.assertTrue(tf.is_tensor(result), "Result should be a TensorFlow tensor.")
        self.assertEqual(result.shape, (2,), "Result should have shape (2,) for 2 measured modes.")
        self.assertEqual(result.dtype, tf.float32, "Result should be float32.")

    def test_run_circuit_once_accepts_numpy_input(self):
        """
        Test that run_circuit_once can accept numpy arrays as input.
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        self.classifier.current_weights = tf.Variable(test_weights, trainable=True, name="test_weights")
        sample_input = np.random.rand(2, 3).astype(np.float32)  # [num_qumodes, num_features]

        # Act
        result = self.classifier.run_circuit_once(sample_input)

        # Assert
        self.assertTrue(tf.is_tensor(result), "Result should be a TensorFlow tensor.")
        self.assertEqual(result.shape, (2,), "Result should have shape (2,) for 2 measured modes.")

    def test_predict_batch_returns_correct_shape(self):
        """
        Test that predict_batch returns a tensor with the correct shape (batch_size,).
        """
        # Arrange
        test_weights = np.random.rand(self.num_weights).astype(np.float32)
        self.classifier.current_weights = tf.Variable(test_weights, trainable=True)
        batch_size = 5
        # Shape: (batch_size, num_qumodes, num_features) -> (5, 2, 3)
        x_batch = tf.random.normal([batch_size, 2, 3])

        # Act
        predictions = self.classifier.predict_batch(x_batch)

        # Assert
        self.assertTrue(tf.is_tensor(predictions), "Predictions should be a TensorFlow tensor.")
        self.assertEqual(predictions.shape, (batch_size,), f"Predictions shape should be ({batch_size},), but was {predictions.shape}.")

    def test_predict_batch_gradient_flow(self):
        """
        Test that gradients flow correctly through predict_batch to all trainable variables.
        """
        # Arrange
        # Use small, non-zero weights to avoid saturation in activation functions
        test_weights = np.full(self.num_weights, 0.1, dtype=np.float32)
        self.classifier.current_weights = tf.Variable(test_weights, trainable=True)
        batch_size = 2
        # Use controlled, small inputs
        x_batch = tf.ones([batch_size, 2, 3], dtype=tf.float32) * 0.1

        with tf.GradientTape() as tape:
            # Act
            predictions = self.classifier.predict_batch(x_batch)
            # Use a dummy loss for gradient calculation
            loss = tf.reduce_sum(predictions)

        # Assert
        trainable_vars = self.classifier.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_vars)
        
        self.assertIsNotNone(gradients, "Gradients should not be None.")
        self.assertEqual(len(gradients), 1, "Should have 1 gradient (weights).")

        grad_map = {var.name: grad for var, grad in zip(trainable_vars, gradients)}

        self.assertIn(self.classifier.current_weights.name, grad_map)

        self.assertIsNotNone(grad_map[self.classifier.current_weights.name], "Gradient for weights should be computed.")

        # Check that gradients are not all zero (which would indicate a problem)
        self.assertTrue(tf.reduce_sum(tf.abs(grad_map[self.classifier.current_weights.name])).numpy() > 1e-6, "Gradient for weights is zero or too small.")


class TestQuantumTrainer(unittest.TestCase):
    """
    Tests for the QuantumTrainer class methods beyond just saving weights.
    """

    def setUp(self):
        """
        Set up a temporary directory and a QuantumTrainer instance for each test.
        """
        self.test_dir = "temp_test_trainer_dir"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        initialize(wires=2, layers=1)
        self.num_weights = 2 * 4 * 1 + 3 # 11 weights

        self.mock_logger = MagicMock()

        # Use a real classifier instance for the spec to get attributes right
        classifier_spec_instance = QuantumClassifier(2, 5)
        self.mock_classifier = MagicMock(spec_set=classifier_spec_instance)

        # The trainer will initialize these weights. We set them to None initially.
        self.mock_classifier.current_weights = None

        # This mock predict_batch is essential for testing gradient flow
        # without running the full quantum circuit.
        def mock_predict_batch(x_batch):
            weights = self.mock_classifier.current_weights
            weights_sum = tf.reduce_sum(weights)
            batch_size = tf.shape(x_batch)[0]
            # Output shape must be (batch_size,) to match the real implementation
            dummy_output = sigmoid(tf.ones(batch_size) * weights_sum * 0.1)
            return dummy_output

        self.mock_classifier.predict_batch.side_effect = mock_predict_batch
        
        # The trainer calls get_trainable_variables on the model.
        # We mock this to be explicit about what's being tracked.
        def get_trainable_variables_mock():
            t_vars = []
            if self.mock_classifier.current_weights is not None and self.mock_classifier.current_weights.trainable:
                t_vars.append(self.mock_classifier.current_weights)
            return t_vars
        
        self.mock_classifier.get_trainable_variables.side_effect = get_trainable_variables_mock

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.initial_weights = np.random.rand(self.num_weights).astype(np.float32)

        self.trainer = QuantumTrainer(
            model=self.mock_classifier,
            optimizer_tf=self.optimizer,
            loss_fn_tf=self.loss_fn,
            logger=self.mock_logger,
            init_weights_val=self.initial_weights,
            save_dir_path=self.test_dir
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """
        Verify that the QuantumTrainer is initialized correctly.
        """
        self.assertIs(self.trainer.model, self.mock_classifier)
        self.assertIs(self.trainer.optimizer, self.optimizer)
        self.assertIs(self.trainer.loss_function, self.loss_fn)
        self.assertTrue(os.path.isdir(self.trainer.checkpoint_dir))
        
        # Check that the trainer initialized the model's weights
        self.assertIsNotNone(self.mock_classifier.current_weights)
        self.assertIsInstance(self.mock_classifier.current_weights, tf.Variable)
        self.assertTrue(self.mock_classifier.current_weights.trainable)
        np.testing.assert_array_equal(self.mock_classifier.current_weights.numpy(), self.initial_weights)

    def test_train_step(self):
        """
        Test a single training step to ensure loss is computed and weights are updated.
        """
        x_batch = tf.random.normal([2, 2, 3])
        y_batch = tf.constant([[1.], [0.]], dtype=tf.float32)
        
        trainable_vars = self.trainer.model.get_trainable_variables()
        initial_vars_numpy = [v.numpy().copy() for v in trainable_vars]

        # Reset mock's call history before the action we are testing
        self.trainer.model.predict_batch.reset_mock()

        # Call the python function directly to avoid tf.function tracing complexities
        loss = self.trainer.train_step.python_function(x_batch, y_batch)

        self.assertTrue(tf.is_tensor(loss))
        self.assertGreater(loss.numpy(), 0)
        
        # Now that we are not in a tf.function context, we can assert the call
        self.trainer.model.predict_batch.assert_called_once_with(x_batch)
        
        updated_vars_numpy = [v.numpy() for v in trainable_vars]
        
        self.assertFalse(
            all(np.allclose(initial, updated) for initial, updated in zip(initial_vars_numpy, updated_vars_numpy)),
            "Trainable variables were not updated by the optimizer."
        )

    def test_validation_step(self):
        """
        Test a single validation step to ensure loss and predictions are returned correctly.
        """
        x_batch = tf.random.normal([2, 2, 3])
        y_batch = tf.constant([[1.], [0.]], dtype=tf.float32)

        # Act
        val_loss, val_predictions = self.trainer.validation_step(x_batch, y_batch)

        # Assert
        # The mock will execute our side_effect function
        expected_predictions = self.trainer.model.predict_batch(x_batch)
        y_batch_flat = tf.reshape(y_batch, [-1])
        expected_loss = self.loss_fn(y_batch_flat, expected_predictions).numpy()
        
        # Calculate AUC for verification
        val_auc = roc_auc_score(y_batch.numpy().flatten(), val_predictions.numpy().flatten())

        # predict_batch was called once in validation_step, and once for our expected_predictions
        self.assertEqual(self.trainer.model.predict_batch.call_count, 2)
        self.assertAlmostEqual(val_loss.numpy(), expected_loss, places=5)
        self.assertIsInstance(val_auc, float)
        self.assertGreaterEqual(val_auc, 0.0)
        self.assertLessEqual(val_auc, 1.0)