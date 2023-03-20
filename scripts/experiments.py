import os
import pickle
import numpy as np
from derivative import dxdt
import matplotlib.pyplot as plt
from keras.optimizers.optimizer_experimental.adamw import AdamW
from utils.log import create_logger
from utils.train import run_one_split, run_lr_split


def run_one():
    batch_size = 100
    debug = False
    task = ('glue', 'mnli')
    prefix = 'experiment'
    # model_config = {'model_checkpoint': 't5-small', 'which_model': 'soft', 'epochs': 0,
    #                 'prompt_transfer': {'model_checkpoint': 't5-small',
    #                 'which_data': ('super_glue', 'boolq')}}
    model_config = {'model_checkpoint': 't5-small', 'which_model': 'soft', 'epochs': 1}
    learning_rate = 0.001
    optimizer_params = {'tag': f'adamw-learning_rate-{learning_rate:.6f}',
                        'optimizer': AdamW(learning_rate=learning_rate)}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints")

    # Create a log object
    logger = create_logger(output_path, filename=f'experiments.log')

    # Run one experiment and log all results
    # If it fails then carry on
    run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                  batch_size=batch_size, cache_path=cache_path, checkpoint_filepath=output_path, debug=debug,
                  prefix=prefix)


def run_few(model_checkpoint, which_model, optimizer_algo, output_path):
    """
    Run a few tasks and return results
    Args:
        model_checkpoint: Model checkpoint to use
        which_model: 'fft' or 'soft'
        optimizer_algo: Parameters for the optimizer
        output_path:

    Returns:

    """
    if 'small' in model_checkpoint:
        batch_size = 100
    else:
        batch_size = 25
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
        result = run_lr_split(logger, model_config=model_config, optimizer_algo=optimizer_algo,
                              which_data=task, batch_size=batch_size, cache_path=cache_path,
                              checkpoint_filepath=output_path, debug=debug,
                              prefix=prefix, force_rerun=True)

        results.append(result)

    return results


def optimizer_checks(model_checkpoint, which_model):
    """

    Returns:
    """
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/optimizer")
    os.makedirs(output_path, exist_ok=True)
    algo = AdamW

    # Learning rate on log scale
    optim_tag = str(algo).split('.')[-1].split("'>")[0]
    # These are the results for this configuration
    filename = os.path.join(output_path, 'lro' + '-' + model_checkpoint + '-' + which_model + '-' + optim_tag + '.p')
    if not os.path.exists(filename):
        results = run_few(model_checkpoint, which_model, algo, output_path)
        with open(filename, 'wb') as outfi:
            pickle.dump(results, outfi)
    else:
        with open(filename, 'rb') as infi:
            results = pickle.load(infi)

    return results


def analyze_results(results):
    plt.figure(figsize=(10, 4.8))
    colors = ['b', 'g', 'r', 'c']
    derivatives = []
    min_der = []
    x_s = []
    loss_s = []
    for cnt, res in enumerate(results):
        x_ = np.array(res['learning_rate'])
        loss = np.array(res['loss'])
        idx = np.logical_and(x_ > 1E-7, x_ < 1)

        x_ = x_[idx]
        loss = loss[idx]

        x_s.append(x_)
        loss_s.append(loss)

        # Find the derivative of multiple points
        # der = dxdt(loss, np.log10(x_), kind='finite_difference', k=3)
        der = dxdt(loss, np.log10(x_), kind='kalman', alpha=0.5)
        derivatives.append(der)
        min_der.append(np.min(der))

    # Make the limit as 20% higher than the minimum derivatives
    # limit = max(min_der) * 0.8
    for cnt, (x_, loss, der) in enumerate(zip(x_s, loss_s, derivatives)):
        plt.subplot(1, 2, 1)
        plt.plot(np.log10(x_), loss, f'-{colors[cnt]}')
        # plt.plot(np.log10(x_[der < limit]), loss[der < limit], f'o{colors[cnt]}')

        plt.subplot(1, 2, 2)
        plt.plot(np.log10(x_), der, colors[cnt])

    plt.ylim([None, 0])

    plt.subplot(1, 2, 1)
    plt.xlabel("Log10 of learning rate")
    plt.ylabel("Training loss")

    plt.subplot(1, 2, 2)
    plt.xlabel("Log10 of learning rate")
    plt.ylabel("derivative of training loss")
    plt.show()


def analyze_one(results):

    for title, result in results.items():
        plt.figure(figsize=(10, 4.8))

        x_ = np.array(result['learning_rate'])
        loss = np.array(result['loss'])

        idx = np.logical_and(x_ > 1E-7, x_ < 1)
        x_ = x_[idx]
        loss = loss[idx]
        min_loss = min(loss)

        # Find the derivative of multiple points
        der = dxdt(loss, np.log10(x_), kind='kalman', alpha=1)
        idx = np.argmin(der)

        optimal_lr = np.log10(x_[idx])
        loss_at_optimal = loss[idx]
        der_at_optimal = der[idx]

        plt.subplot(1, 2, 1)
        plt.plot(np.log10(x_), loss)
        plt.plot([optimal_lr, optimal_lr], [loss_at_optimal, min_loss], '-k')

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
        plt.suptitle(title + '-lr-' + f'{10**optimal_lr:.2e}')
    plt.show()


def analyze_all(results):

    res_asdict = {f'{cnt}': v for cnt, v in enumerate(results)}
    analyze_one(res_asdict)


if __name__ == '__main__':
    # run_one()
    ress = optimizer_checks('t5-small', 'fft')
    # analyze_results(ress)
    analyze_all(ress)
    # print(os.getcwd())
