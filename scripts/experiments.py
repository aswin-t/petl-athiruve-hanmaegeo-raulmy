import os
import pickle
import numpy as np
from utils.log import create_logger
from utils.train import run_one_split
from keras.optimizers import SGD, Adam
from keras.optimizers.optimizer_experimental.adamw import AdamW


def run_one():
    batch_size = 10
    debug = False
    task = ('super_glue', 'boolq')
    prefix = 'experiment'
    # model_config = {'model_checkpoint': 't5-small', 'which_model': 'soft', 'epochs': 0,
    #                 'prompt_transfer': {'model_checkpoint': 't5-small',
    #                 'which_data': ('super_glue', 'boolq')}}
    model_config = {'model_checkpoint': 't5-small', 'which_model': 'soft', 'epochs': 1}
    optimizer_params = {'algo': 'adam', 'params': {'learning_rate': 0.01}}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints")

    # Create a log object
    logger = create_logger(output_path, filename=f'experiments.log')

    # Run one experiment and log all results
    # If it fails then carry on
    run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                  batch_size=batch_size, cache_path=cache_path, checkpoint_filepath=output_path, debug=debug,
                  prefix=prefix)


def run_few(model_checkpoint, which_model, optimizer_params, output_path):
    """
    Run a few tasks and return results
    Args:
        model_checkpoint: Model checkpoint to use
        which_model: 'fft' or 'soft'
        optimizer_params: Parameters for the optimizer
        output_path:

    Returns:

    """
    batch_size = 10
    debug = False
    tasks = [('super_glue', 'multirc'), ('super_glue', 'rte'), ('glue', 'cola'), ('glue', 'qnli')]
    prefix = 'optimizer'
    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': 1}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    logger = create_logger(output_path, filename=f'optimizer_experiments.log')

    # Run one experiment and log all results
    # If it fails then carry on
    results = []
    for task in tasks:
        result = run_one_split(
            logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
            batch_size=batch_size, cache_path=cache_path, checkpoint_filepath=output_path, debug=debug,
            prefix=prefix, force_rerun=True)

        results.append(result)

    return results


def optimizer_checks(model_checkpoint, which_model):
    """

    Returns:
    """
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/optimizer")
    os.makedirs(output_path, exist_ok=True)

    # Learning rate on log scale
    for algo in [AdamW, SGD, Adam]:
        for learning_rate in 10**np.linspace(-5, -1, 10):
            # Create a tag for the optimizer
            optim_tag = str(algo).split('.')[-1].split("'>")[0]
            optim_tag += f'-learning_rate-{learning_rate:.05f}'

            # Create an optimizer object
            optimi_ = algo(learning_rate=learning_rate)
            tag = model_checkpoint + '-' + which_model + '-' + optim_tag

            # These are the results for this configuration
            filename = os.path.join(output_path, tag + '.p')
            if os.path.exists(filename):
                continue
            else:
                optimizer_params = {'optimizer': optimi_, 'tag': optim_tag}
                results = run_few(model_checkpoint, which_model, optimizer_params, output_path)

                with open(filename, 'wb') as outfi:
                    pickle.dump(results, outfi)


if __name__ == '__main__':
    # run_one()
    optimizer_checks('t5-small', 'fft')
