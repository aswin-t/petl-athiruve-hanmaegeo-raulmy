import os
import pickle
import numpy as np
import tensorflow as tf
from derivative import dxdt
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers.models.t5 import TFT5ForConditionalGeneration
from keras.optimizers.optimizer_experimental.adamw import AdamW
from utils import constants
from utils.constants import Tasks
from utils.data import PrepDataset
from utils.log import create_logger
from utils.train import run_one_split, run_lr_split, get_optimizer


def run_one():
    batch_size = 100
    debug = False
    task = ('super_glue', 'multirc')
    prefix = 'experiment'
    learning_rate = 2E-4/100  # sst2 - 1E-5  # rte - 1.5E-3
    epochs = 30

    model_config = {'model_checkpoint': 't5-small', 'which_model': 'fft', 'epochs': 10}
    scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate, t_mul=2, first_decay_steps=int(epochs/3)*25, m_mul=0.1)

    optimizer_params = {'tag': f'adamw-learning_rate-{learning_rate:.6f}',
                        'optimizer': AdamW(learning_rate=learning_rate)}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/experiments")
    os.makedirs(output_path, exist_ok=True)

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
                              prefix=prefix, force_run=True)

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


def get_training_samples(model_checkpoint):
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints/batches")
    os.makedirs(output_path, exist_ok=True)

    # Create a log object
    logger = create_logger(output_path, filename=f'experiments.log')

    dprep = PrepDataset(logger=logger, checkpoint=model_checkpoint)

    results = {}
    for benchmark in ['superglue', ]:
        tasks = Tasks()[benchmark]
        results[benchmark] = {}
        for task in tasks:
            _, _, counts = dprep.load(which=task, batch_size=1, cache_path=cache_path)
            print(f"{task}: {counts['train']}")
            results[benchmark][task] = counts['train']

    with open('sizes.p', 'wb') as outfi:
        pickle.dump(results, outfi)


def analyze_all(results):

    res_asdict = {f'{cnt}': v for cnt, v in enumerate(results)}
    analyze_one(res_asdict)


def soft_experiment(model_checkpoint='t5-small', max_batch_size=100, min_num_batches=50, benchmark='glue', epochs=30,
                    gpu=0):
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
    prefix = 'experiment'
    which_model = 'soft'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': epochs}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    model_checkpoint = model_config['model_checkpoint']
    which_model = model_config['which_model']
    logger = create_logger(checkpoint_filepath, filename=f'{prefix}-{model_checkpoint}-{which_model}-benchmark.log')
    logger.info(f'Performing {benchmark} tuning')

    # Ensure at least 50 batches
    # task = ('super_glue', 'rte')
    task = ('glue', 'rte')
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task]/min_num_batches))}

    # for lr in [1E-7, 1E-6, 3E-6, 1E-5, 1E-4, 1E-3, 1E-3, 1E-1, 1, 10]:
    for lr in [0.3, ]:
        # Get the batch size
        # Run one experiment and log all results
        # If it fails then carry on
        optimizer_lrs = {task: lr}
        optimizer_params = get_optimizer(optimizer_lrs, which_data=task)
        run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                      batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                      debug=True, prefix=prefix)


def token():

    checkpoint = "google/t5-base-lm-adapt"
    # Get the tokenizer for this data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.replace('_-_', '/'),
                                              model_max_length=constants.ENCODER_MAX_LEN)
    text = 'summarize: There is so much work, I am behind on my HW, lectures and my office work.'
    intoken = tokenizer(text, return_tensors='tf')
    model = TFT5ForConditionalGeneration.from_pretrained(checkpoint)
    out_tokens = model.generate(intoken['input_ids'].numpy().tolist(), max_length=50)
    print(out_tokens)
    print(tokenizer.decode(out_tokens.numpy().reshape(-1, ), skip_special_tokens=False))


if __name__ == '__main__':
    # mcp = 'liangtaiwan/t5-v1_1-lm100k-base'.replace('/', '_-_')
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    # mcp = 't5-base'.replace('/', '_-_')
    soft_experiment(model_checkpoint=mcp, max_batch_size=25, min_num_batches=50, benchmark='glue', epochs=100, gpu=0)
    # ress = optimizer_checks('t5-small', 'fft')
    # analyze_results(ress)
    # analyze_all(ress)
    # print(os.getcwd())
    # token()
