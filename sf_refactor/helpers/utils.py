import h5py
import numpy as np
import tensorflow as tf

# DATA IS ALREADY SORTED BY PT (highest to lowest)
def load_data(path, max_jets, wires):
    with h5py.File(path, "r") as f:
        jet_constituents = f["jetConstituentsList"][:max_jets, :wires, :]     # (N,4,3)
        truth_labels = f["truth_labels"][:max_jets].astype(np.float32)     # (N,)
    return tf.convert_to_tensor(jet_constituents, tf.float32), tf.convert_to_tensor(truth_labels, tf.float32)

def get_loss_fn(photons, label, shift_sigmoid=None, tanh = False, loss_type="bce", dim_cutoff=None):
    """
    Returns a function loss_fn(y_true, logit) so the training loop
    doesn't need to care whether we are using BCE (logits) or
    MSE (probabilities).
    """
    from_logits = False

    y_true  = tf.expand_dims(label, 0)          # shape (1,)
    logit   = tf.reduce_mean(photons) # used to use reduce_sum but mean makes more sense
    if dim_cutoff is not None:
        logit /= dim_cutoff  # allow the states to have up to 10 photons

    if loss_type.lower() == "bce":
        if shift_sigmoid is not None:
            logit -= shift_sigmoid  # shift the sigmoid to the left
            from_logits = True
            prob = tf.sigmoid(logit)
        elif tanh:
            logit = tf.math.tanh(logit)
            from_logits = False
            prob = (logit + 1.0) / 2.0
        else:
            from_logits = True
            prob = tf.sigmoid(logit)

        y_logit = tf.expand_dims(logit, 0)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits) # Applies sigmoid internally

        def _loss(y_true, logit):
            return bce(y_true, logit)

    elif loss_type.lower() == "mse":
        y_logit = tf.expand_dims(logit, 0)
        mse = tf.keras.losses.MeanSquaredError()

        # MSE should see probabilities in [0,1]
        def _loss(y_true, logit):
            return mse(y_true, tf.sigmoid(logit))

        prob = tf.sigmoid(logit)

    else:
        raise ValueError(f"Unknown loss type: {loss_type} (use 'bce' or 'mse')")
    
    loss_fn = _loss
    
    loss = loss_fn(y_true, y_logit)

    return loss, prob 