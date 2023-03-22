import os
import pickle
from utils.log import create_logger
from utils.train import run_benchmark


def run_model(benchmark, model_config: dict, optimizer_lrs: dict, checkpoint_filepath: str, debug: bool = False,
              prefix='', batch_size=None):
    """

    Args:
         model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
                'epochs': <Optional>
        benchmark:
        checkpoint_filepath: Location of where the checkpoints are stored
        optimizer_lrs: optimizer learning rates for each dataset
        debug: if True then eager model of evaluation is run, else graph mode
        prefix: Prefix to add to the model names
        batch_size: Training batch size
    Returns:

    """

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    batch_size = 100 if batch_size is None else batch_size

    # Create a log object
    model_checkpoint = model_config['model_checkpoint']
    which_model = model_config['which_model']
    logger = create_logger(checkpoint_filepath, filename=f'{prefix}-{model_checkpoint}-{which_model}-benchmark.log')
    logger.info(f'Performing {benchmark} tuning')

    #  Run the superglue benchmnark
    run_benchmark(logger, model_config=model_config, optimizer_lrs=optimizer_lrs, batch_size=batch_size,
                  cache_path=cache_path, checkpoint_filepath=checkpoint_filepath, debug=debug, benchmark=benchmark,
                  one_task=None, prefix=prefix)


def run_fft():
    model_checkpoint = 't5-small'
    which_model = 'fft'
    benchmark = 'target'
    batch_size = 100
    epochs = 30
    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': epochs}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")

    # Benchmark of target signifies target tasks
    # Learning rate on log scale
    try:
        all_tasks_tag = model_checkpoint + '-' + which_model + '-' + benchmark
        filepath = os.path.join(checkpoint_filepath, 'optimizer/lro-' + all_tasks_tag + '.p')
        with open(filepath, 'rb') as infi:
            optimizer_lrs = pickle.load(infi)
    except FileNotFoundError:
        raise FileNotFoundError('Was optimization run to get learning rates?')

    # Scale the learning rate by the number of epochs * number of batches per epoch
    optimizer_lrs = {k: v for k, v in optimizer_lrs['fine_tuning'].items()}

    # Benchmark can be given as this tuple of atsks or a benchmark name such as 'glue' or 'super_glue'
    run_model(benchmark=benchmark, model_config=model_config, optimizer_lrs=optimizer_lrs, debug=False,
              prefix='athiruve', batch_size=batch_size, checkpoint_filepath=checkpoint_filepath)


if __name__ == '__main__':
    run_fft()
