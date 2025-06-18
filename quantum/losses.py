import pennylane.numpy as np
import sys,tqdm
import quantum.math_functions as mfunc
import tensorflow as tf

def sigmoid(x):
    return 1/(1+np.exp(-x))

def semi_classical_cost(weights,inputs=None,quantum_circuit=None,return_fid=False):
    if return_fid:
        fid=quantum_circuit(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return 100.*np.array(cost),100.*np.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_circuit(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid#np.array([-quantum_circuit(weights,item) for item in inputs]).mean()
    return 100.*np.array(cost,requires_grad=False)

def batch_semi_classical_cost(weights,inputs=None,quantum_circuit=None,return_fid=False):
    if return_fid:
        fid=quantum_circuit(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return np.array(100.*cost,requires_grad=True).mean(),np.array(100.*fid,requires_grad=True).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_circuit(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid
    batched_average_cost=100.*(np.array(cost,requires_grad=True).mean())#np.array([-quantum_circuit(weights,item) for item in inputs]).mean()
    return batched_average_cost

def classifier_cost(weights, inputs=None, quantum_circuit=None, labels=None, return_scores=False, loss_type='MSE', reg=1.):
    """
    Computes the cost function for the classifier.

    This function is designed for a TensorFlow-based workflow.
    It calculates the loss based on the expectation values from the quantum circuit.
    """
    exp_vals = quantum_circuit(weights, inputs)  # The quantum circuit now returns a tensor of shape (batch_size,)
    
    if loss_type == 'BCE':
        # Apply sigmoid for binary cross-entropy
        scores = tf.nn.sigmoid(exp_vals)
        loss_fn = binary_crossentropy(labels, scores)
    elif loss_type == 'MSE':
        # Use expectation values directly for mean squared error
        scores = exp_vals
        loss_fn = mean_squared_error(labels, scores)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    if return_scores:
        return loss_fn, scores
    return loss_fn

def probabilistic_loss(weights, inputs=None, quantum_circuit=None, labels=None, return_scores=False, loss_type='BCE'):
    """
    Computes a probabilistic loss for a quantum circuit that outputs probabilities.
    """
    batch_size = tf.shape(labels)[0]
    # quantum_circuit is expected to return probabilities, shape (batch_size, num_classes)
    probs = quantum_circuit(weights, inputs)
    
    # Gather the probabilities corresponding to the true labels
    indices = tf.stack([tf.range(batch_size), labels], axis=1)
    scores = 1.0 - tf.gather_nd(probs, indices)
    
    loss_fn = tf.reduce_mean(scores)
        
    if return_scores:
        return loss_fn, scores
    return loss_fn

def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error between true labels and predictions.

    Args:
        y_true (tf.Tensor): The true labels.
        y_pred (tf.Tensor): The predicted values.

    Returns:
        tf.Tensor: The mean squared error.
    """
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    """
    Calculates the binary cross-entropy between true labels and predictions.

    Args:
        y_true (tf.Tensor): The true labels.
        y_pred (tf.Tensor): The predicted values.

    Returns:
        tf.Tensor: The binary cross-entropy loss.
    """
    return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
