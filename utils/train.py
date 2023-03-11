# Credit:https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb
import os
import tensorflow as tf
from typing import Union
from utils.data import PrepDataset
from utils.metric import evaluate_metric
from utils.model import PETLSoftPrompt, FullFineTune

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


def _get_model(which_model, checkpoint, debug):
    """

    Args:
        which_model: Which model to use, FullFineTune or SoftPrompt or ...
        checkpoint: Which model checkpoint to use
        debug: If debug is True then model is run in eager model otherwise in graph mode

    Returns:

    """

    if which_model.lower() in ['sp', 'soft_prompt', 'softprompt', 'soft', 'petl']:

        # learning_rate = 0.001  # Instead set a static learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        # Create a model instance
        model = PETLSoftPrompt.from_pretrained(checkpoint)

        # This makes the embedding layer non-trainable
        # The layer is called shared because it is shared between the encoder and decoder
        model.shared.trainable = False

        # We want the soft prompt to be trainable but all other weights must not be trainable
        for b in model.encoder.block:
            b.trainable = False
        model.encoder.final_layer_norm.trainable = False

        # We don't want any trainable parameters in the decode layer
        model.layers[2].trainable = False

    elif which_model.lower in ['full', 'full_fine_tune', 'fullfinetune', 'fft']:
        # learning_rate = 0.001  # Instead set a static learning rate
        optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=0.1)

        # Create a model instance
        model = FullFineTune.from_pretrained(checkpoint)

    else:
        raise KeyError(f'Model {which_model} is not supported')

    # Compile the model with Categorical accuracy metric
    model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy'), ],
                  run_eagerly=debug)

    return model


def main_flow(checkpoint: str = "t5-small", which_data: Union[str, tuple] = 'squad', which_model: str = 'fft',
              batch_size: int = 4, epochs: int = 10, cache_path: str = None, debug: bool = False):

    # Prepare the Dataset
    dprep = PrepDataset(checkpoint=checkpoint)
    tfsplits, splits, counts = dprep.load(which=which_data, batch_size=batch_size, cache_path=cache_path,
                                          as_batches=False)

    # Get the model from the
    model = _get_model(which_model, checkpoint, debug)
    model.summary()

    # Ready to train the model
    model.fit(tfsplits['train'], epochs=epochs, callbacks=[], validation_data=tfsplits['val'], initial_epoch=0)
    # results = evaluate_metric(which_data, checkpoint, model, splits['val'])
    return model


if __name__ == '__main__':
    cp = os.path.join(os.path.dirname(__file__), "../cache")
    model_o = main_flow(checkpoint='t5-small', which_data=('super_glue', 'boolq'), which_model='soft', batch_size=100,
                        cache_path=cp, debug=False, epochs=1)
