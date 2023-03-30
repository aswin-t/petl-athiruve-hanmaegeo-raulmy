import os
import pickle
import tensorflow as tf
from utils import constants
from utils.constants import Tasks
from utils.log import create_logger
from utils.train import run_benchmark

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


def run_model(benchmark, model_config: dict, optimizer_params: dict, checkpoint_filepath: str, debug: bool = False,
              prefix='', batch_size=None, epochs=None):
    """

    Args:
         model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
        epochs: Number training epochs
        benchmark:
        checkpoint_filepath: Location of where the checkpoints are stored
        optimizer_params: Optimizer parameters for each dataset
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

    #  Run the superglue benchmark
    run_benchmark(logger, model_config=model_config, optimizer_params=optimizer_params, batch_size=batch_size,
                  cache_path=cache_path, checkpoint_filepath=checkpoint_filepath, debug=debug, benchmark=benchmark,
                  one_task=None, prefix=prefix, epochs=epochs)


def run_fft(model_checkpoint='t5-small', max_batch_size=100, min_num_batches=50, benchmark='target', epochs=30, gpu=0):
    """

    Args:
        model_checkpoint: t5-small, t5-base
        max_batch_size: Maximum batch size
        min_num_batches: Minimum number of batches
        benchmark: glue, super_glue, target
        epochs: Number of training epochs
        gpu: Which GPU to use

    Returns:
    """
    which_model = 'fft'
    target_steps = 20000

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': epochs}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")

    # Ensure at least 50 batches
    tasks = Tasks()[benchmark]
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task] / min_num_batches)) for task in tasks}

    # Maintaining approximately the same number of steps for all datasets
    # epochs = target_specs/steps per epoch
    if epochs is None:
        epochs = {task:  int(target_steps/(constants.COUNTS[task]/batch_size[task])) for task in tasks}
    else:
        epochs = {task: epochs for task in tasks}

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
    run_model(benchmark=benchmark, model_config=model_config, optimizer_params=optimizer_lrs, debug=False,
              prefix='athiruve', batch_size=batch_size, checkpoint_filepath=checkpoint_filepath, epochs=epochs)


def run_soft(model_checkpoint='t5-small', max_batch_size=100, min_num_batches=50, benchmark='glue', epochs=None, gpu=0):
    """

    Args:
        model_checkpoint: t5-small, t5-base
        max_batch_size: Maximum batch size
        min_num_batches: Minimum number of batches
        benchmark: glue, super_glue, target
        epochs: Number of training epochs
        gpu: Which GPU to use

    Returns:
    """

    which_model = 'soft'
    target_steps = 30000
    prefix = 'baseline'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")

    # Ensure at least 50 batches
    tasks = Tasks()[benchmark]
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task]/min_num_batches)) for task in tasks}

    # Maintaining approximately the same number of steps for all datasets
    # epochs = target_specs/steps per epoch
    if epochs is None:
        epochs = {task:  int(target_steps/(constants.COUNTS[task]/batch_size[task])) for task in tasks}
    else:
        epochs = {task: epochs for task in tasks}

    # Benchmark of target signifies target tasks
    optimizer_params = {task: {'learning_rate': 0.1, 'weight_decay': 1E-3} for task in tasks}

    # Benchmark can be given as this tuple of atsks or a benchmark name such as 'glue' or 'super_glue'
    run_model(benchmark=benchmark, model_config=model_config, optimizer_params=optimizer_params, debug=False,
              prefix=prefix, batch_size=batch_size, checkpoint_filepath=checkpoint_filepath, epochs=epochs)


if __name__ == '__main__':
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(benchmark='super_glue', gpu=1, epochs=None, model_checkpoint=model_checkpoint_, max_batch_size=32)
