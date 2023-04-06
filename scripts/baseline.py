import os
import math
import pickle
import tensorflow as tf
from utils import constants
from utils.constants import Tasks
from utils.log import create_logger
from utils.train import run_benchmark

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


def run_model(benchmark, model_config: dict, optimizer_params: dict, checkpoint_filepath: str, debug: bool = False,
              prefix='', batch_size=None, epochs=None, token_equalize=False, force_run: bool = False):
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
        token_equalize: Equalize token lengths
        force_run: Force the run even if it is previously completed
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
                  one_task=None, prefix=prefix, epochs=epochs, token_equalize=token_equalize,
                  force_run=force_run)


def run_fft(model_checkpoint='t5-small', batch_size=32, benchmark='target', epochs=None, token_equalize=False,
            prefix='baseline_fft', gpu=0, force_run: bool = False):
    """

    Args:
        model_checkpoint: t5-small, t5-base
        batch_size: Mini batch size
        benchmark: glue, super_glue, target
        epochs: Number of training epochs
        gpu: Which GPU to use
        prefix: Model prefix to use
        token_equalize: Equalize token lengths
        force_run: Force the run or not
    Returns:
    """
    which_model = 'fft'
    target_steps = 30000

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../mycheckpoints")

    if isinstance(benchmark, list):
        tasks = benchmark
    elif isinstance(benchmark, tuple):
        raise TypeError('Benchmark can be string, glue, sueprglue or a list of tuples of task names')
    else:
        tasks = Tasks()[benchmark]

    # Maintaining approximately the same number of steps for all datasets
    # epochs = target_specs/steps per epoch
    if epochs is None:
        epochs = {task:  math.ceil(target_steps/(constants.COUNTS[task]/batch_size)) for task in tasks}
    else:
        epochs = {task: epochs for task in tasks}

    # Benchmark of target signifies target tasks
    # Learning rate on log scale
    try:
        all_tasks_tag = model_checkpoint + '-' + which_model + '-' + tasks[0][0]
        filepath = os.path.join(checkpoint_filepath, 'optimizer/lro-' + all_tasks_tag + '.p')
        with open(filepath, 'rb') as infi:
            optimizer_lrs = pickle.load(infi)
        optimizer_lrs = {k: v for k, v in optimizer_lrs['fine_tuning'].items()}
    except FileNotFoundError:
        # optimizer_lrs = {task: 1E-6 for task in tasks}
        raise FileNotFoundError('Was optimization run to get learning rates?')

    # Benchmark of target signifies target tasks
    optimizer_params = {task: {'learning_rate': optimizer_lrs[task], 'weight_decay': 1E-4,
                               'beta_1': 0.8, 'beta_2': 0.999} for task in tasks}

    # Benchmark can be given as this tuple of atsks or a benchmark name such as 'glue' or 'super_glue'
    run_model(benchmark=benchmark, model_config=model_config, optimizer_params=optimizer_params, debug=False,
              prefix=prefix, batch_size=batch_size, checkpoint_filepath=checkpoint_filepath, epochs=epochs,
              token_equalize=token_equalize, force_run=force_run)


def run_soft(model_checkpoint='t5-small', batch_size=32, benchmark='glue', epochs=None, token_equalize=False,
             prefix='baseline_soft', gpu=0, force_run: bool = False):
    """

    Args:
        model_checkpoint: t5-small, t5-base
        batch_size: Mini batch size
        benchmark: glue, super_glue, target
        epochs: Number of training epochs
        gpu: Which GPU to use
        prefix: Prefix for run differentiate log files
        token_equalize: Equalize token lengths
        force_run: Force the run or not
    Returns:
    """

    which_model = 'soft'
    target_steps = 30000
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../mycheckpoints")

    if isinstance(benchmark, list):
        tasks = benchmark
    elif isinstance(benchmark, tuple):
        raise TypeError('Benchmark can be string, glue, sueprglue or a list of tuples of task names')
    else:
        tasks = Tasks()[benchmark]

    # Maintaining approximately the same number of steps for all datasets
    # epochs = target_specs/steps per epoch
    if epochs is None:
        epochs = {task:  math.ceil(target_steps/(constants.COUNTS[task]/batch_size)) for task in tasks}
    else:
        epochs = {task: epochs for task in tasks}
    # Benchmark of target signifies target tasks
    optimizer_params = {task: {'learning_rate': 0.3, 'weight_decay': 1E-4, 'beta_1': 0.8, 'beta_2': 0.999}
                        for task in tasks}

    # Benchmark can be given as this tuple of atsks or a benchmark name such as 'glue' or 'super_glue'
    run_model(benchmark=benchmark, model_config=model_config, optimizer_params=optimizer_params, debug=False,
              prefix=prefix, batch_size=batch_size, checkpoint_filepath=checkpoint_filepath, epochs=epochs,
              token_equalize=token_equalize, force_run=force_run)


if __name__ == '__main__':
    constants.SEED = 42
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(model_checkpoint=model_checkpoint_, batch_size=32, benchmark='glue', prefix='baseline_soft_equal_2',
             token_equalize=True, gpu=0, force_run=True)
