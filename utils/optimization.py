import os
import pickle
import numpy as np
from derivative import dxdt
import matplotlib.pyplot as plt
from keras.optimizers.optimizer_experimental.adamw import AdamW
from utils.log import create_logger
from utils.train import run_lr_split
from utils.constants import get_tasks


def run_one(model_checkpoint, which_model, which_data, optimizer_algo, output_path):
    """
    Run a few tasks and return results
    Args:
        model_checkpoint: Model checkpoint to use
        which_model: 'fft' or 'soft'
        which_data:
        optimizer_algo: Parameters for the optimizer
        output_path:

    Returns:

    """
    if 'small' in model_checkpoint:
        batch_size = 100
    else:
        batch_size = 25

    debug = False
    prefix = 'optimizer'
    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': 1}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    logger = create_logger(output_path, filename=f'optimizer_experiments.log')

    # Run one experiment and log all results
    # If it fails then carry on
    result = run_lr_split(logger, model_config=model_config, optimizer_algo=optimizer_algo,
                          which_data=which_data, batch_size=batch_size, cache_path=cache_path,
                          checkpoint_filepath=output_path, debug=debug,
                          prefix=prefix, force_rerun=True)

    return result


def analyze_results(results, output_path):
    """

    Args:
        results:
        output_path:

    Returns:

    """

    lrs = {}
    for title, (result, task) in results.items():
        plt.figure(figsize=(10, 4.8))

        x_ = np.array(result['learning_rate'])
        loss = np.array(result['loss'])

        idx = x_ < 1
        x_ = x_[idx]
        loss = loss[idx]

        # Find the derivative of multiple points
        # der = dxdt(loss, np.log10(x_), kind="savitzky_golay", left=.5, right=.5, order=3)
        done = False
        idx = 0
        der = []
        while not done:
            try:
                der = dxdt(loss, np.log10(x_), kind="kalman", alpha=0.5)
                idx = np.argmin(der)
                done = True
            except Exception:
                x_ = x_[:-1]
                loss = loss[:-1]

        optimal_lr = np.log10(x_[idx])
        loss_at_optimal = loss[idx]
        der_at_optimal = der[idx]

        plt.subplot(1, 2, 1)
        plt.plot(np.log10(x_), loss)
        plt.plot([optimal_lr, optimal_lr], [loss_at_optimal, 0], '-k')

        plt.subplot(1, 2, 2)
        plt.plot(np.log10(x_), der)
        plt.plot([optimal_lr, optimal_lr], [der_at_optimal, 0], '-k')
        plt.ylim([None, 0])

        plt.subplot(1, 2, 1)
        plt.xlabel("Log10 of learning rate")
        plt.ylabel("Training loss")

        plt.subplot(1, 2, 2)
        plt.xlabel("Log10 of learning rate")
        plt.ylabel("derivative of training loss")
        plt.suptitle(title + '-' + f'{10**optimal_lr:.6f}')

        lrs[task] = 10**optimal_lr

        filepath = os.path.join(output_path, 'lro-' + title + '.png')
        plt.savefig(filepath)

    plt.show()
    return lrs


def get_adamw_lrs(model_checkpoint, which_model, benchmark):
    """

    Returns:
    """
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/optimizer")
    os.makedirs(output_path, exist_ok=True)
    optimizer_algo = AdamW

    # Learning rate on log scale
    all_tasks_tag = model_checkpoint + '-' + which_model + '-' + benchmark
    filepath_all = os.path.join(output_path, 'lro-data-' + all_tasks_tag + '.p')

    # Check if all tasks optimization has been performed in the past
    if not os.path.exists(filepath_all):
        # If some or all tasks are due

        # Get the list of tasks
        tasks = get_tasks(benchmark)

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
                result = run_one(model_checkpoint, which_model, task, optimizer_algo, output_path)
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


if __name__ == '__main__':
    ress = get_adamw_lrs('t5-small', 'fft', benchmark='target')
    print(ress)
