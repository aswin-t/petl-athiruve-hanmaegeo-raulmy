import os
import pickle
import numpy as np
from derivative import dxdt
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.optimizers.optimizer_experimental.adamw import AdamW
# from keras.optimizers.optimizer_experimental.adafactor import Adafactor
from utils.log import create_logger
from utils import constants
from utils.constants import Tasks
from utils.train import run_lr_split, get_model_official_name, create_prompt_tag


def run_one(logger, model_checkpoint, which_model, which_data, optimizer_algo, output_path, batch_size):
    """
    Run a few tasks and return results
    Args:
        logger: Logger object
        model_checkpoint: Model checkpoint to use
        which_model: 'fft', 'soft', 'spt'
        which_data:
        optimizer_algo: Parameters for the optimizer
        batch_size:
        output_path:

    Returns:

    """

    debug = True
    prefix = 'optimizer'
    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': 1}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Run one experiment and log all results
    # If it fails then carry on
    result = run_lr_split(logger, model_config=model_config, optimizer_algo=optimizer_algo,
                          which_data=which_data, batch_size=batch_size, cache_path=cache_path,
                          checkpoint_filepath=output_path, debug=debug,
                          prefix=prefix, force_run=True)

    return result


def run_one_spt(logger, model_checkpoint, which_data, source_config, optimizer_algo, output_path, batch_size):
    """
    Run a few tasks and return results
    Args:
        logger: Logger object
        model_checkpoint: Model checkpoint to use
        source_config:
        which_data: Which data
        optimizer_algo: Parameters for the optimizer
        batch_size:
        output_path:

    Returns:

    """

    debug = False
    prefix = 'optimizer'
    model_config = {'model_checkpoint': model_checkpoint, 'which_model': 'spt', 'epochs': 1,
                    'prompt_transfer': source_config}
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Run one experiment and log all results
    # If it fails then carry on
    result = run_lr_split(logger, model_config=model_config, optimizer_algo=optimizer_algo,
                          which_data=which_data, batch_size=batch_size, cache_path=cache_path,
                          checkpoint_filepath=output_path, debug=debug,
                          prefix=prefix, force_run=True)

    return result


def _fine_tuning_lr(x_, loss, der, idx):
    """

    Args:
        x_:
        loss:
        der:
        idx:

    Returns:

    """

    x_ = x_[:idx + 1]
    loss = loss[:idx + 1]
    der = der[:idx + 1]

    # Index the lr lower than 0
    idx = np.where(der > np.min(der)/100)[0][-1]
    der = der[idx + 1:]
    x_ = x_[idx + 1:]
    loss = loss[idx + 1:]

    # Fine tuning LR
    optimal_ft_lr = np.mean(np.log10(x_))
    loss_at_optimal_lr = loss[np.sum(x_ < 10 ** optimal_ft_lr)]
    der_at_optimal_lr = der[np.sum(x_ < 10 ** optimal_ft_lr)]

    return x_, loss, der, optimal_ft_lr, loss_at_optimal_lr, der_at_optimal_lr


def _get_metrics(loss, x_, fine_tuning=False):
    # Find the derivative of multiple points
    done = False
    idx = 0
    der = []
    while not done:
        # noinspection PyBroadException
        try:
            # der = dxdt(loss, np.log10(x_), kind="kalman", alpha=0.5)
            der = dxdt(loss, np.log10(x_), kind="savitzky_golay", left=.5, right=.5, order=1)
            idx = np.argmin(der)
            done = True
        except Exception:
            x_ = x_[:-1]
            loss = loss[:-1]

    optimal_lr = np.log10(x_[idx])
    loss_at_optimal = loss[idx]
    der_at_optimal = der[idx]

    if not fine_tuning:
        return x_, loss, der, optimal_lr, loss_at_optimal, der_at_optimal
    else:
        x__, loss_, der_, optimal_ft_lr, loss_at_optimal_, der_at_optimal_ = _fine_tuning_lr(x_, loss, der, idx)
        return x__, loss_, der_, optimal_ft_lr, loss_at_optimal_, der_at_optimal_


def analyze_results(results, output_path, lower_range=5E-7, upper_range=10):
    """

    Args:
        results:
        output_path:
        lower_range:
        upper_range:

    Returns:

    """

    lrs = {'training': {}, 'fine_tuning': {}}
    for title, (result, task) in results.items():
        plt.figure(figsize=(10, 4.8))

        xin = np.array(result['learning_rate'])[10:]
        loss_in = np.array(result['loss'])[10:]

        idx = np.logical_and(lower_range < xin, xin < upper_range)
        # This is for fine tuning, the learning rates can be lower
        x_, loss, der, optimal_lr, loss_at_optimal, der_at_optimal = \
            _get_metrics(loss_in[idx], xin[idx], fine_tuning=False)

        # This is for training, the learning rates can be lower
        x__, loss_, der_, optimal_ft_lr, loss_at_optimal_, der_at_optimal_ = \
            _get_metrics(loss_in[idx], xin[idx], fine_tuning=True)

        plt.subplot(1, 2, 1)
        plt.plot(np.log10(x_), loss)
        plt.plot(np.log10(x__), loss_)
        plt.plot([optimal_lr, optimal_lr], [loss_at_optimal, 0], '-k')
        plt.plot([optimal_ft_lr, optimal_ft_lr], [loss_at_optimal_, 0], '-r')
        plt.ylim([np.min(loss) * 0.8, 1.4 * max(np.min(loss), loss_at_optimal)])

        plt.subplot(1, 2, 2)
        plt.plot(np.log10(x_), der)
        plt.plot(np.log10(x__), der_)
        plt.plot([optimal_lr, optimal_lr], [der_at_optimal, 0], '-k')
        plt.plot([optimal_ft_lr, optimal_ft_lr], [der_at_optimal_, 0], '-r')
        plt.ylim([1.2 * np.min(der), 0])

        plt.subplot(1, 2, 1)
        plt.xlabel("Log10 of learning rate")
        plt.ylabel("Training loss")

        optimal_lr = 10 ** optimal_lr
        optimal_ft_lr = 10 ** optimal_ft_lr

        lrs['training'][task] = optimal_lr
        lrs['fine_tuning'][task] = optimal_ft_lr

        plt.subplot(1, 2, 2)
        plt.xlabel("Log10 of learning rate")
        plt.ylabel("derivative of training loss")
        plt.suptitle(title + '-' + f'{optimal_lr:.8f} - {optimal_ft_lr:.8f} - ratio {optimal_lr / optimal_ft_lr:0f}')

        filepath = os.path.join(output_path, 'lro-' + title + '.png')
        plt.savefig(filepath)

    plt.show()
    return lrs


def get_adamw_spt_lrs(model_checkpoint, which_model, source_config, batch_size, lower_range=5E-7, upper_range=10):
    """

    Returns:
    """
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/optimizer")
    os.makedirs(output_path, exist_ok=True)
    optimizer_algo = AdamW

    # Create a log object
    logger = create_logger(output_path, filename=f'optimizer_experiments.log')

    # Learning rate on log scale
    prompt_official_name = get_model_official_name(source_config['which_model'])
    prompt_tag = create_prompt_tag(source_config['model_checkpoint'], prompt_official_name, source_config['which_data'])
    all_tasks_tag = model_checkpoint + '-' + which_model + '-' + prompt_tag
    filepath_all = os.path.join(output_path, 'lro-data-' + all_tasks_tag + '.p')

    # Check if all tasks optimization has been performed in the past
    if not os.path.exists(filepath_all):
        # If some or all tasks are due

        # Get the list of tasks
        tasks = Tasks()['target']

        results = {}
        for task in tasks:
            # These are the results for this configuration
            task_tag = "".join(f"{x}-" for x in task)
            dict_key = model_checkpoint + '-' + which_model + '-' + task_tag
            filename = 'lro' + '-' + dict_key + '.p'

            # has this one been run before?
            filepath = os.path.join(output_path, filename)
            if not os.path.exists(filepath):

                # Results from running one lr optimization loop
                result = run_one_spt(logger, model_checkpoint, task, source_config, optimizer_algo, output_path,
                                     batch_size)
                with open(filepath, 'wb') as outfi:
                    pickle.dump(result, outfi)
            else:
                with open(filepath, 'rb') as infi:
                    result = pickle.load(infi)

            results[dict_key] = (result, task)

        # If all tasks have completed then just that one file
        with open(filepath_all, 'wb') as outfi:
            pickle.dump(results, outfi)
    else:
        with open(filepath_all, 'rb') as infi:
            results = pickle.load(infi)

    # Now analyze the results
    lrs = analyze_results(results, output_path)
    filepath_all = os.path.join(output_path, 'lro-' + all_tasks_tag + '.p')

    # If all tasks have completed then just that one file
    with open(filepath_all, 'wb') as outfi:
        pickle.dump(lrs, outfi)

    return lrs


def get_adamw_lrs(model_checkpoint, which_model, benchmark, max_batch_size=100, min_num_batches=50,
                  lower_range=5E-7, upper_range=10):
    """

    Returns:
    """

    # Ensure at least 50 batches
    if isinstance(benchmark, str):
        tasks = Tasks()[benchmark]
        bm_str = benchmark
    else:
        tasks = benchmark
        bm_str = ''.join(f"{x}-" for x in benchmark[0])
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task] / min_num_batches)) for task in tasks}

    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/optimizer")
    os.makedirs(output_path, exist_ok=True)

    use_adam = False
    if use_adam:
        optimizer_algo = AdamW
    else:
        optimizer_algo = SGD

    # Create a log object
    logger = create_logger(output_path, filename=f'optimizer_experiments-soft.log')

    # Learning rate on log scale
    all_tasks_tag = model_checkpoint + '-' + which_model + '-' + bm_str
    filepath_all = os.path.join(output_path, 'lro-data-' + all_tasks_tag + '.p')

    # Check if all tasks optimization has been performed in the past
    if not os.path.exists(filepath_all):
        # If some or all tasks are due

        results = {}
        for task in tasks:
            # These are the results for this configuration
            task_tag = "".join(f"{x}-" for x in task)
            dict_key = model_checkpoint + '-' + which_model + '-' + task_tag
            filename = 'lro' + '-' + dict_key + '.p'

            # has this one been run before?
            filepath = os.path.join(output_path, filename)
            if not os.path.exists(filepath):

                # Results from running one lr optimization loop
                result = run_one(logger, model_checkpoint, which_model, task, optimizer_algo, output_path,
                                 batch_size[task])
                with open(filepath, 'wb') as outfi:
                    pickle.dump(result, outfi)
            else:
                with open(filepath, 'rb') as infi:
                    result = pickle.load(infi)

            results[dict_key] = (result, task)

        # If all tasks have completed then just that one file
        with open(filepath_all, 'wb') as outfi:
            pickle.dump(results, outfi)
    else:
        with open(filepath_all, 'rb') as infi:
            results = pickle.load(infi)

    # Now analyze the results
    lrs = analyze_results(results, output_path, lower_range=lower_range, upper_range=upper_range)
    filepath_all = os.path.join(output_path, 'lro-' + all_tasks_tag + '.p')

    # If all tasks have completed then just that one file
    with open(filepath_all, 'wb') as outfi:
        pickle.dump(lrs, outfi)

    return lrs


if __name__ == '__main__':
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    bm = (('glue', 'mrpc'), )
    get_adamw_lrs(model_checkpoint=mcp, which_model='soft', benchmark=bm, max_batch_size=25,
                  min_num_batches=50, lower_range=1E-6, upper_range=10)
