import h5py
import numpy as np
import tensorflow as tf

# DATA IS ALREADY SORTED BY PT (highest to lowest)
def load_data(path, max_jets, wires):
    with h5py.File(path, "r") as f:
        jet_constituents = f["jetConstituentsList"][:max_jets, :wires, :]     # (N,4,3)
        truth_labels = f["truth_labels"][:max_jets].astype(np.float32)     # (N,)
    return tf.convert_to_tensor(jet_constituents, tf.float32), tf.convert_to_tensor(truth_labels, tf.float32)

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