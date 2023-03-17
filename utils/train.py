# Credit:https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb
import os
import re
import glob
import tensorflow as tf
from typing import Union
from utils.data import PrepDataset
from utils.log import create_logger
from utils.metric import evaluate_metric
from utils.model import get_model, model_history_to_dlog

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


def check_config(config, key, default, required):
    """

    Args:
        config:
        key:
        default:
        required:

    Returns:

    """
    try:
        # Found the key and return it
        tmp = config[key]
        del config[key]
        return tmp
    except KeyError:
        if required:
            raise KeyError(f'Model config requires {key}')
        else:
            return default


def load_checkpoint(tag: str, checkpoint_dir: str, load_best: bool = False):
    """

    Args:
        tag: Filename tag to add
        checkpoint_dir: Checkpoint folder to
        load_best: Load the best model or simply the latest epoch
    Returns:

    """

    strg_chk = re.compile(r'-e(\d+)-v(\d*\.\d*).hdf5')

    # Find all filenames that match the current tag
    cur_epoch = -1
    cur_val = 1E100

    # Empty filename
    filen = ''

    # Look for all files with the tag
    filenames = glob.glob(os.path.join(checkpoint_dir, tag) + '*')
    if filenames:
        for filename in filenames:
            # This should match the expected value
            pat = strg_chk.search(filename)
            epoch = int(pat.group(1))
            val = float(pat.group(2))

            if load_best and val < cur_val:
                filen = filename
                cur_epoch = epoch
                cur_val = val
            elif not load_best and epoch > cur_epoch:
                filen = filename
                cur_epoch = epoch
                cur_val = val

    return filen, cur_epoch


def _create_file_tag(model_checkpoint, which_model, which_data, epochs, optimizer_params):
    """

    Returns: A tag for this unique model configuration

    """
    tag = model_checkpoint + '-' + which_model + '-'
    tag += "".join(f'{x}-' for x in which_data)
    tag += f'epochs-{epochs}-'
    tag += f'{optimizer_params["algo"]}-'
    tag += "".join(f'{k}-{v}-' for k, v in optimizer_params['params'].items())

    return tag[:-1]


def run_one_split(logger, model_config: dict = None, optimizer_params: dict = None,
                  which_data: Union[str, tuple] = 'squad',
                  batch_size: int = 4, cache_path: str = None, output_path: str = None,
                  debug: bool = False, prefix=''):
    """

    Args:
        logger: Object of python logging class
        model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'librabry' -> library of soft prompts
                'epochs': <Optional>
        optimizer_params:
        which_data: Which benchmark data source we are fine-tuning the data to ('squad', ), ('super_glue', 'boolq'), ..
        batch_size: Number of rows to use per batch
        cache_path: Path to store the cache files
        output_path: Path to store the model checkpoints and log file
        debug: if True then eager model of evaluation is run, else graph mode
        prefix: Prefix to add to the model names

    Returns:

    """

    # 1. Get and process inputs
    # Get all the inputs for the model
    model_checkpoint = check_config(model_config, 'model_checkpoint', default=None, required=True)
    which_model = check_config(model_config, 'which_model', default=None, required=True)
    epochs = check_config(model_config, 'epochs', default=10, required=False)
    if model_config:
        raise KeyError(f'Unexpected keys {list(model_config.keys())} in model_config')

    default = {'algo': 'adam', 'params': {'learning_rate': 0.001}}
    optimizer_params = default if optimizer_params is None else optimizer_params

    # 2. Create a unique tag and load model checkpoint
    # Create a tag for this unique model
    tag = _create_file_tag(model_checkpoint, which_model, which_data, epochs, optimizer_params)
    if prefix:
        tag = prefix + '-' + tag
    filen, start_epoch = load_checkpoint(tag, output_path, load_best=False)

    if start_epoch >= epochs:
        print('Model was previously run with equal or more epochs and completed. No need to run again')
        return True

    logger.info(f'This evaluation tag is {tag}')

    # 3. Prepare the data
    # Prepare the Dataset
    dprep = PrepDataset(logger=logger, checkpoint=model_checkpoint)
    tfsplits, splits, counts = dprep.load(which=which_data, batch_size=batch_size, cache_path=cache_path)

    # 4. Get the model
    # Get the model from the
    model, official_name = get_model(which_model, model_checkpoint, debug, optimizer_params, logger, filen)
    model.summary()

    # 5. Train the model
    checkpoint_filepath = os.path.join(output_path, tag + '-e{epoch:02d}-v{val_accuracy:.3f}.hdf5')
    model_checkpoint_callback = \
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True)
    history = model.fit(tfsplits['train'], epochs=epochs, callbacks=[model_checkpoint_callback, ],
                        validation_data=tfsplits['val'], initial_epoch=start_epoch)

    # Save history and metrics
    model_history_to_dlog(logger, history.history, official_name)
    results = evaluate_metric(logger, tag, which_data, model_checkpoint, model, splits['test'])

    return results


def run_benchmark(model_config: dict = None, optimizer_params: dict = None, batch_size: int = 4, cache_path: str = None,
                  output_path: str = None, debug: bool = False, benchmark='superglue',
                  one_task: str = None, prefix=''):
    """

    Args:
        model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'librabry' -> library of soft prompts
                'epochs': <Optional>
        optimizer_params:
        batch_size: Number of rows to use per batch
        cache_path: Path to store the cache files
        output_path: Path to store the model checkpoints and log file
        benchmark: Which benchmark to run glue or superglue
        debug: if True then eager model of evaluation is run, else graph mode
        one_task: Which superglue task to run
        prefix: Prefix to add to the model names

    Returns:

    """
    one_task = '' if one_task is None else one_task

    # These are the superglue tasks that we want to perform
    if benchmark == 'glue':
        tasks = (('glue', 'cola'), ('glue', 'mrpc'), ('glue', 'qnli'), ('glue', 'qqp'),
                 ('glue', 'rte'), ('glue', 'sst2'), ('glue', 'wnli'), ('glue', 'stsb'))
    elif benchmark == 'superglue':
        tasks = (('super_glue', 'boolq'), ('super_glue', 'rte'), ('super_glue', 'wic'), ('super_glue', 'wsc.fixed'),
                 ('super_glue', 'multirc'), ('super_glue', 'cb'), ('super_glue', 'copa'))
    else:
        raise KeyError(f'Benchmark {benchmark} is not supported')

    # Check of the one task is
    if one_task:
        for task in tasks:
            if task == one_task:
                break
    else:
        raise KeyError(f'Task {one_task} is not in benchmark {benchmark}')

    # Create a log object
    logger = create_logger(output_path, filename=f'{prefix}model_super_glue.log')
    logger.info(f'Performing super_glue tuning on {prefix}')

    # For each task run the model
    for task in tasks:
        # If only one specific task must be run then run that
        if one_task:
            if task != one_task:
                continue

        try:
            # Run one experiment and log all results
            # If it fails then carry on
            run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                          batch_size=batch_size, cache_path=cache_path, output_path=output_path, debug=debug,
                          prefix=prefix)
        except Exception as e:
            # Capture the exception and
            logger.exception(e)
            logger.warning('Exception was raised')


def run_model(model_config: dict = None, debug: bool = False, prefix=''):
    """

    Args:
         model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
                'epochs': <Optional>
        debug: if True then eager model of evaluation is run, else graph mode
        prefix: Prefix to add to the model names
    Returns:

    """

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints")
    optimizer_params = {'algo': 'adam', 'params': {'learning_rate': 0.0001}}
    batch_size = 100

    #  Run the superglue benchmnark
    run_benchmark(model_config=model_config, optimizer_params=optimizer_params, batch_size=batch_size,
                  cache_path=cache_path, output_path=output_path, debug=debug,benchmark='superglue',
                  one_task=None, prefix=prefix)

    #  Run the superglue benchmnark
    run_benchmark(model_config=model_config, optimizer_params=optimizer_params, batch_size=batch_size,
                  cache_path=cache_path, output_path=output_path, debug=debug, benchmark='glue',
                  one_task=None, prefix=prefix)


if __name__ == '__main__':
    prefixo = 'aswin-'
    # Run this model and collect results in log file
    model_configo = {'model_checkpoint': 't5-small', 'which_model': 'fft', 'epochs': 30}
    run_model(model_config=model_configo, debug=False, prefix=prefixo)

    # Run this model and collect results in log file
    model_configo = {'model_checkpoint': 't5-base', 'which_model': 'fft', 'epochs': 30}
    run_model(model_config=model_configo, debug=False, prefix=prefixo)

    # Run this model and collect results in log file
    model_configo = {'model_checkpoint': 't5-large', 'which_model': 'fft', 'epochs': 30}
    run_model(model_config=model_configo, debug=False, prefix=prefixo)

