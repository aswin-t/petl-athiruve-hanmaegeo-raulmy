# Credit:https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb
import os
import gc
import re
import copy
import glob
import time
import tensorflow as tf
from typing import Union
from keras.optimizers.optimizer_experimental.adamw import AdamW
from utils.constants import Tasks
from utils.data import PrepDataset
from utils.metric import evaluate_metric
from utils.model import get_model, model_history_to_dlog, PromptCallback, LinearRampScheduler, BatchLossCallback

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


def _load_checkpoint(tag: str, checkpoint_dir: str, epochs: Union[int, None], load_best: bool = False,
                     force_run: bool = False):
    """

    Args:
        tag: Filename tag to add
        checkpoint_dir: Checkpoint folder to
        load_best: Load the best model or simply the latest epoch
        force_run: Force the run or not
    Returns:

    """

    strg_chk = re.compile(r'-e(\d+)-v(\d*\.\d*)\.(hdf5|npy)')

    # Find all filenames that match the current tag
    cur_epoch = -1
    cur_val = 0

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

            if load_best and val > cur_val:
                filen = filename
                cur_epoch = epoch
                cur_val = val
            elif not load_best and epoch > cur_epoch:
                filen = filename
                cur_epoch = epoch
                cur_val = val

    if epochs is not None and cur_epoch >= epochs - 1:
        print('Model was previously run with equal or more epochs and completed. No need to run again')
        if not force_run:
            return None
        else:
            filen = ''
            cur_epoch = -1

    cur_epoch += 1
    return filen, cur_epoch, filenames


def _create_file_tag(model_checkpoint, which_model, which_data, optimizer_tag):
    """

    Returns: A tag for this unique model configuration

    """
    tag = model_checkpoint + '-' + which_model + '-'
    tag += "".join(f'{x}-' for x in which_data)
    tag += optimizer_tag
    return tag


def create_prompt_tag(model_checkpoint, model_name, which_data):
    """

    Returns: A tag for this unique model configuration

    """
    tag = model_checkpoint + '-' + model_name + '-'
    tag += "".join(f'{x}-' for x in which_data)
    return tag[:-1]


def _save_soft_prompt(model, which_model, checkpoint_filepath, model_checkpoint, which_data):
    """
    Save the prompt

    Args:
        model: Fitted model
        model_checkpoint: Checkpoint from which the model was loaded
        checkpoint_filepath: Filepath to store the prompts
        which_model: Prompts are only saved for the soft model
        which_data: Which data was fit for the model

    Returns:

    """

    # This is a model we want to save the prompts for
    if which_model == 'PETLSoftPrompt':
        filepath = os.path.join(checkpoint_filepath, 'soft_prompt')

        #  Make the folder if it does not exist
        os.makedirs(filepath, exist_ok=True)

        # Only the model checkpoint
        prompt_tag = create_prompt_tag(model_checkpoint, which_model, which_data)
        model.save_prompt(os.path.join(filepath, 'soft-prompt-' + prompt_tag))

        return prompt_tag
    else:
        return ""


def _load_soft_prompt(model, which_model, prompt_model, checkpoint_filepath, model_checkpoint, which_data):
    """
    Save the prompt

    Args:
        model: Fitted model
        model_checkpoint: Checkpoint from which the model was loaded
        checkpoint_filepath: Filepath to store the prompts
        which_model: Prompts are only saved for the soft model
        prompt_model: Model for which teh prompt is to be laoded
        which_data: Which data was fit for the model
    Returns:

    """

    # This is a model we want to save the prompts for
    if which_model in ['PETLSoftPrompt', 'PETLSoftPromptTransfer']:
        # Soft prompt is not requested
        if not (model_checkpoint and which_data):
            return ""

        #  Make the folder if it does not exist
        os.makedirs(checkpoint_filepath, exist_ok=True)

        # Only the model checkpoint
        prompt_tag = create_prompt_tag(model_checkpoint, prompt_model, which_data)
        model.load_prompt(os.path.join(checkpoint_filepath, 'soft-prompt-' + prompt_tag))

        return prompt_tag
    else:
        return ""


def _remove_unwanted_checkpoint_files(logger, tag, checkpoint_filepath):
    """
    For
    Args:
        logger: Logger object
        tag: Filename tag to
        checkpoint_filepath: Location of files

    Returns:

    """
    # We need only two files:
    # a. The final epoch, in case we want to continue
    # b. The best validation score for model evaluations
    filen_last, _, all_files = _load_checkpoint(tag, checkpoint_filepath, epochs=None)
    filen_best, _, _ = _load_checkpoint(tag, checkpoint_filepath, load_best=True, epochs=None)
    logger.info(f'Best model file is {filen_best}, last epoch file is {filen_last}')

    # Remove these files
    remove_files = [x for x in all_files if x not in [filen_last, filen_best]]
    for fname in remove_files:
        logger.info(f'Removing unwanted file {fname}')
        if os.path.isfile(fname):
            os.remove(fname)
    return True


def _get_checkpoint_callback(official_name, checkpoint_filepath, tag):
    """

    Args:
        official_name: Official name of the model
        checkpoint_filepath: Path to save checkpoint
        tag: Model tag

    Returns:

    """

    if official_name == 'FullFineTune':
        filepath = os.path.join(checkpoint_filepath, tag + '-e{epoch:02d}-v{val_accuracy:.3f}.hdf5')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=True)
    elif official_name == 'PETLSoftPrompt':
        filepath = os.path.join(checkpoint_filepath, tag)
        model_checkpoint_callback = PromptCallback(filepath=filepath, best_is_lower=False)
    else:
        raise NotImplementedError(f'Callback for model type {official_name} is not supported')

    return model_checkpoint_callback


def get_model_official_name(which_model):
    """

    Args:
        which_model:

    Returns:

    """
    if which_model.lower() in ['sp', 'soft_prompt', 'softprompt', 'soft', 'petl']:
        model_name = 'PETLSoftPrompt'
    elif which_model.lower() in ['spt', 'soft_prompt_transfer', 'softprompttransfer']:
        model_name = 'PETLSoftPromptTransfer'
    elif which_model.lower() in ['full', 'full_fine_tune', 'fullfinetune', 'fft']:
        model_name = 'FullFineTune'
    else:
        raise NotImplementedError(f'Model {which_model} is not supported')
    return model_name


def _get_config(model_config):
    model_checkpoint = check_config(model_config, 'model_checkpoint', default=None, required=True)
    which_model = check_config(model_config, 'which_model', default=None, required=True)
    epochs = check_config(model_config, 'epochs', default=10, required=False)
    prompt_specs = check_config(model_config, 'prompt_transfer', default=None, required=False)

    if prompt_specs is not None:
        prompt_model_checkpoint = prompt_specs['model_checkpoint']
        prompt_which_data = prompt_specs['which_data']
        prompt_which_model = prompt_specs['which_model']
    else:
        prompt_model_checkpoint = ''
        prompt_which_data = ''
        prompt_which_model = ''

    if model_config:
        raise KeyError(f'Unexpected keys {list(model_config.keys())} in model_config')

    return model_checkpoint, which_model, epochs, prompt_model_checkpoint, prompt_which_data, prompt_which_model


def run_lr_split(logger, optimizer_algo, model_config: dict = None,
                 which_data: Union[str, tuple] = 'squad', batch_size: int = 4, cache_path: str = None,
                 checkpoint_filepath: str = None, debug: bool = False, prefix='', force_run: bool = False):
    """

    Args:
        logger: Object of python logging class
        optimizer_algo: Optimizer class such as SGD or ADAM
        model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
                'epochs': <Optional>
        which_data: Which benchmark data source we are fine-tuning the data to ('squad', ), ('super_glue', 'boolq'), ..
        batch_size: Number of rows to use per batch
        cache_path: Path to store the cache files
        checkpoint_filepath: Path to store the model checkpoints and log file
        debug: if True then eager model of evaluation is run, else graph mode
        prefix: Prefix to add to the model names
        force_run: Force a rerun even if a file exists

    Returns:

    """
    gc.enable()

    model_config = model_config.copy()

    # 1. Get and process inputs
    # Get inputs from model config
    model_checkpoint, which_model, epochs, prompt_model_checkpoint, prompt_which_data, prompt_which_model = \
        _get_config(model_config)
    logger.info(f'LR optimization on checkpoint {model_checkpoint} of {prompt_which_model}')

    # Get model official name
    official_name = get_model_official_name(which_model)

    # 2. Create a unique tag and load model checkpoint
    # Create a tag for this unique model
    tag = _create_file_tag(model_checkpoint, official_name, which_data, "")
    if prefix:
        tag = prefix + '-' + tag

    # Is there a model that has already been created for this?
    filen, start_epoch, _ = _load_checkpoint(tag, checkpoint_filepath, epochs, load_best=False,
                                             force_run=force_run)

    # Running this evaluation
    logger.info(f'This evaluation tag is {tag}')

    # 3. Prepare the data
    # Load teh data to memory
    dprep = PrepDataset(logger=logger, checkpoint=model_checkpoint)
    tfsplits, splits, counts = dprep.load(which=which_data, batch_size=batch_size, cache_path=cache_path)
    _log_gpu_usage(logger, prefix="Dataset")

    # Increase the learning rate linearly within one training epoch
    learning_scheduler = LinearRampScheduler(initial_learning_rate=1E-7, final_learning_rate=10,
                                             total_steps=int(counts['train'] / batch_size))
    optimizer = optimizer_algo(learning_rate=learning_scheduler)

    # 4. Get the model
    # Load the appropriate model
    model = get_model(official_name, model_checkpoint, debug, optimizer, logger, filen)
    if prompt_model_checkpoint:
        prompt_official_name = get_model_official_name(prompt_which_model)
        prompt_checkpoint_filepath = os.path.join(checkpoint_filepath, "..")
        # Load the soft prompt for this model
        prompt_tag = _load_soft_prompt(model, official_name, prompt_official_name, prompt_checkpoint_filepath,
                                       prompt_model_checkpoint, prompt_which_data)
        tag += '-softprompt-' + prompt_tag

    model.summary()
    _log_gpu_usage(logger, prefix="Model created")

    # 5. Train the model
    model_optimizer_callback = BatchLossCallback(logger=logger)
    model.fit(tfsplits['train'], epochs=epochs, callbacks=[model_optimizer_callback, ],
              validation_data=tfsplits['val'], initial_epoch=start_epoch)
    _log_gpu_usage(logger, prefix="Model fit")

    # Delete the model
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(20)
    _log_gpu_usage(logger, prefix="return")

    return copy.deepcopy(model_optimizer_callback.history)


def _log_gpu_usage(logger, prefix):
    """

    Returns:

    """

    # Check the memory devices
    gpu_devices = tf.config.get_visible_devices('GPU')
    gpu_usage_str = ""
    for cnt, _ in enumerate(gpu_devices):
        usage = tf.config.experimental.get_memory_info(f"GPU:{cnt}")['current']
        gpu_usage_str += f'GPU: {cnt} Usage: {usage},'
    logger.info(f"{prefix}: {gpu_usage_str[:-1]}")


def run_one_split(logger, model_config: dict = None, optimizer_params=None,
                  which_data: Union[str, tuple] = 'squad', batch_size: int = 4, cache_path: str = None,
                  checkpoint_filepath: str = None, debug: bool = False, prefix='', force_run: bool = False):
    """

    Args:
        logger: Object of python logging class
        model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
                'epochs': <Optional>
        optimizer_params: {'optimizer': <Object of optimizer class or None to use default>, 'tag': <text description>}
        which_data: Which benchmark data source we are fine-tuning the data to ('squad', ), ('super_glue', 'boolq'), ..
        batch_size: Number of rows to use per batch
        cache_path: Path to store the cache files
        checkpoint_filepath: Path to store the model checkpoints and log file
        debug: if True then eager model of evaluation is run, else graph mode
        prefix: Prefix to add to the model names
        force_run: Force a rerun even if a file exists

    Returns:

    """
    model_config = model_config.copy()

    # 1. Get and process inputs
    # Get inputs from model config
    model_checkpoint, which_model, epochs, prompt_model_checkpoint, prompt_which_data, prompt_which_model = \
        _get_config(model_config)

    # Get the optimizer specifications
    optimizer_params = {} if optimizer_params is None else optimizer_params
    optimizer = check_config(optimizer_params, 'optimizer', default=None, required=False)
    optimizer_tag = check_config(optimizer_params, 'tag', default='adam-learning_rate-0.001', required=False)

    # Get model official name
    official_name = get_model_official_name(which_model)

    # 2. Create a unique tag and load model checkpoint
    # Create a tag for this unique model
    tag = _create_file_tag(model_checkpoint, official_name, which_data, optimizer_tag)
    if prefix:
        tag = prefix + '-' + tag

    # Is there a model that has already been created for this?
    ret = _load_checkpoint(tag, checkpoint_filepath, epochs, load_best=False, force_run=force_run)
    if ret is None:
        _remove_unwanted_checkpoint_files(logger, tag, checkpoint_filepath)
        return True
    else:
        (filen, start_epoch, filenames) = ret

    # Running this evaluation
    logger.info(f'This evaluation tag is {tag}')

    # 3. Prepare the data
    # Load the data to memory
    dprep = PrepDataset(logger=logger, checkpoint=model_checkpoint)
    add_taskname = True if official_name == 'FullFineTune' else False
    tfsplits, splits, counts, led = dprep.load(which=which_data, batch_size=batch_size, cache_path=cache_path,
                                               add_taskname=add_taskname)
    _log_gpu_usage(logger, prefix="Dataset")

    # 4. Get the model
    # Load the appropriate model
    model = get_model(official_name, model_checkpoint, debug, optimizer, logger, filen, dprep, led)

    # 4.2 Maybe we want to load an existing prompt
    if prompt_model_checkpoint and not filen:
        prompt_official_name = get_model_official_name(which_model)
        filepath = os.path.join(checkpoint_filepath, 'soft_prompt')
        # Load the soft prompt for this model
        prompt_tag = _load_soft_prompt(model, official_name, prompt_official_name, filepath, prompt_model_checkpoint,
                                       prompt_which_data)
        tag += '-softprompt-' + prompt_tag

    # Display model summary
    model.summary()
    _log_gpu_usage(logger, prefix="Model created")

    # 5. Train the model
    model_checkpoint_callback = _get_checkpoint_callback(official_name, checkpoint_filepath, tag)
    history = model.fit(tfsplits['train'], epochs=epochs, callbacks=[model_checkpoint_callback, ],
                        validation_data=tfsplits['val'], initial_epoch=start_epoch)
    _log_gpu_usage(logger, prefix="Model fit")

    if history.history:
        # Save history and metrics
        model_history_to_dlog(logger, history.history, official_name)
        history = history.history
    else:
        history = None

    # Delete the model
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(15)

    # Log the GPU usage
    _log_gpu_usage(logger, prefix="Model cleared")

    # 6. Evaluate metric
    # For evaluating the test metric, load the best model
    filen_best, start_epoch, all_files = \
        _load_checkpoint(tag, checkpoint_filepath, load_best=True, epochs=None)
    model = get_model(official_name, model_checkpoint, debug, optimizer, logger, filen_best)
    results = evaluate_metric(logger, tag, which_data, model_checkpoint, model, splits['val'])
    _log_gpu_usage(logger, prefix="Model evaluated")

    # Append history before returning
    results['history'] = history

    # Save the soft prompts if this is a soft prompt model
    _save_soft_prompt(model, official_name, checkpoint_filepath, model_checkpoint, which_data)

    # Delete the model
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(5)

    # Done with everything, so remove all unwanted files
    _remove_unwanted_checkpoint_files(logger, tag, checkpoint_filepath)

    # Log GPU usage after everything is done
    _log_gpu_usage(logger, prefix="return")
    return results


def get_optimizer(optimizer_lrs, which_data):
    """

    Args:
        optimizer_lrs:
        which_data:

    Returns:

    """

    optimizer = AdamW(optimizer_lrs[which_data])
    optimizer_tag = f'adamw-learning_rate-{optimizer_lrs[which_data]}'
    return {'optimizer': optimizer, 'tag': optimizer_tag}


def run_benchmark(logger, model_config: dict = None, optimizer_lrs=None, batch_size: Union[int, dict] = 4,
                  cache_path: str = None, checkpoint_filepath: str = None, debug: bool = False, benchmark='superglue',
                  one_task: str = None, prefix=''):
    """

    Args:
        logger: Logger object
        model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
                'epochs': <Optional>
        optimizer_lrs: Dict with key as which_data as value as learning rate
        batch_size: Number of rows to use per batch
        cache_path: Path to store the cache files
        checkpoint_filepath: Path to store the model checkpoints and log file
        benchmark: Which benchmark to run glue or superglue
        debug: if True then eager model of evaluation is run, else graph mode
        one_task: Which superglue task to run
        prefix: Prefix to add to the model names

    Returns:

    """

    one_task = '' if one_task is None else one_task
    prefix = prefix + '-' if prefix else prefix

    # These are the superglue tasks that we want to perform
    tasks = Tasks()[benchmark]

    # Convert batch size into a dictionary to pick a value for each task
    if isinstance(batch_size, int):
        batch_size = {k: batch_size for k in tasks}

    # Check of the one task is
    if one_task:
        for task in tasks:
            if task == one_task:
                break
        else:
            raise KeyError(f'Task {one_task} is not in benchmark {benchmark}')

    # For each task run the model
    for task in tasks:
        # If only one specific task must be run then run that
        if one_task:
            if task != one_task:
                continue

        # Get the optimizer params and then run model
        optimizer_params = get_optimizer(optimizer_lrs, which_data=task)
        try:
            # Get the batch size
            # Run one experiment and log all results
            # If it fails then carry on
            run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                          batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                          debug=debug, prefix=prefix)
        except Exception as e:
            # Capture the exception and
            logger.exception(e)
            logger.warning('Exception was raised')
