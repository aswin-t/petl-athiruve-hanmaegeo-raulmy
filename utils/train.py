# Credit:https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb
import os
from typing import Union
from model import get_model
from utils.data import PrepDataset
from utils.metric import evaluate_metric

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


def run_one_split(checkpoint: str = "t5-small", which_data: Union[str, tuple] = 'squad', which_model: str = 'fft',
                  batch_size: int = 4, epochs: int = 10, cache_path: str = None, debug: bool = False):
    """

    Args:
        checkpoint: Which model checkpoint to use
        which_data: Which benchmark data source we are fine-tuning the data to ('squad', ), ('super_glue', 'boolq'), ..
        which_model: If 'fft' then full fine-tuning of all parameters. If 'soft' then onyl soft prompt is tuned
        batch_size: Number of rows to use per batch
        epochs: Number of epochs to train
        cache_path: Path to store the cache files
        debug: if True then eager model of evaluation is run, else graph mode

    Returns:

    """

    # Get the model from the
    model = get_model(which_model, checkpoint, debug)
    model.summary()

    # Prepare the Dataset
    dprep = PrepDataset(checkpoint=checkpoint)
    tfsplits, splits, counts = dprep.load(which=which_data, batch_size=batch_size, cache_path=cache_path,
                                          as_batches=False, model=model)

    # Ready to train the model
    model.fit(tfsplits['train'], epochs=epochs, callbacks=[], validation_data=tfsplits['val'], initial_epoch=0)

    results = evaluate_metric(which_data, checkpoint, model, splits['val'])
    return model


if __name__ == '__main__':
    cp = os.path.join(os.path.dirname(__file__), "../cache")
    model_o = run_one_split(checkpoint='t5-small', which_data=('super_glue', 'boolq'), which_model='soft',
                            batch_size=100, cache_path=cp, debug=False, epochs=1)
