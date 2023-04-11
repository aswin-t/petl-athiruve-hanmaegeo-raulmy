import os
import math
import pickle
import random
import tensorflow as tf
from itertools import product
from transformers import AutoTokenizer
from transformers.models.t5 import TFT5ForConditionalGeneration
from utils import constants
from utils.constants import Tasks
from utils.data import PrepDataset
from utils.log import create_logger
from utils.train import run_one_split, get_optimizer

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


def get_training_samples(model_checkpoint):
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../mycheckpoints/batches")
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


def _create_optimizer_experiments(total_steps):
    """

    Returns:
    """

    scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts
    weight_decays = [1E-6, 1E-4, 1E-2]
    learning_rates = [0.1, 0.3, 0.7, 1.0]
    beta1_decays = [0.7, 0.8, 0.9, 0.99]
    beta2_decays = [0.99, 0.999, 0.9999]
    do_schedulers = [False, True]

    first_decay_steps = int(total_steps / 3)

    combos = list(product(*[learning_rates, weight_decays, do_schedulers, beta1_decays, beta2_decays]))
    random.shuffle(combos)

    out = []
    for combo in combos:
        # Get the learning rate
        if combo[2]:
            lr = scheduler(initial_learning_rate=combo[0], first_decay_steps=first_decay_steps, t_mul=2.0)
        else:
            lr = combo[0]
        param = {'learning_rate': lr, 'weight_decay': combo[1], 'beta_1': combo[3], 'beta_2': combo[4]}
        out.append(param)

    return out


def _cosine_similarity(tensor_a, tensor_b):

    # Element wise multiplication of the two arrays
    element_mult = tf.math.multiply(tensor_a, tensor_b)
    dot_product = tf.math.reduce_sum(element_mult, axis=-1)

    tensor_a_norm = tf.norm(tensor_a, axis=-1)
    tensor_b_norm = tf.norm(tensor_b, axis=-1)

    magnitude = tf.math.multiply(tensor_a_norm, tensor_b_norm)
    token_similarity = tf.math.divide(dot_product, magnitude)

    return token_similarity


def hyperparameter(prefix='hyperparameter', model_checkpoint='t5-small', batch_size=32, task=None, gpu=0,
                   target_steps=20000):
    """

    Args:
        prefix:element
        model_checkpoint: t5-small, t5-base
        batch_size: Maximum batch size
        gpu: Which GPU to use
        task:
        target_steps:

    Returns:
    """
    which_model = 'soft'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../mycheckpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    model_checkpoint = model_config['model_checkpoint']
    which_model = model_config['which_model']
    logger = create_logger(checkpoint_filepath, filename=f'{prefix}-{model_checkpoint}-{which_model}-benchmark.log')
    logger.info(f'Performing {task} tuning')

    # Ensure at least 50 batches
    batch_size = {task: batch_size}

    # Get the batch size
    # Run one experiment and log all results
    # If it fails then carry on
    epochs = math.ceil(target_steps / (constants.COUNTS[task] / batch_size[task]))
    total_steps = int(epochs * (constants.COUNTS[task] / batch_size[task]))
    optimizer_experiments = _create_optimizer_experiments(total_steps)

    for optimizer_param in optimizer_experiments:
        optimizer_param_ = get_optimizer(optimizer_param)
        run_one_split(logger, model_config=model_config, optimizer_params=optimizer_param_, which_data=task,
                      batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                      debug=True, prefix=prefix, epochs=epochs)


def experiment(prefix='experiment', model_checkpoint='t5-small', batch_size=32, task=None,
               epochs=None, gpu=0, optimizer_param=None, encoder_max_length=None, token_equalize=False,
               which_model='soft', force_run: bool = False):
    """

    Args:
        prefix:
        model_checkpoint: t5-small, t5-base
        batch_size: Batch size for experiment
        epochs: Number of training epochs
        gpu: Which GPU to use
        task:
        optimizer_param:
        encoder_max_length: Max length to encode the inputs
        token_equalize: Equalize token size
        which_model:
        force_run:

    Returns:
    """
    target_steps = 30000

    default = {'learning_rate': 0.1, 'weight_decay': 1E-3}
    optimizer_param = default if optimizer_param is None else optimizer_param

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model,
                    'encoder_max_length': encoder_max_length}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../mycheckpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    model_checkpoint = model_config['model_checkpoint']
    which_model = model_config['which_model']
    logger = create_logger(checkpoint_filepath, filename=f'{prefix}-{model_checkpoint}-{which_model}-benchmark.log')
    logger.info(f'Performing {task} tuning')

    if epochs is None:
        epochs = math.ceil(target_steps/(constants.COUNTS[task]/batch_size))

    # Get the batch size
    # Run one experiment and log all results
    # If it fails then carry on
    optimizer_param_ = get_optimizer(optimizer_param)
    run_one_split(logger, model_config=model_config, optimizer_params=optimizer_param_, which_data=task,
                  batch_size=batch_size, cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                  debug=True, prefix=prefix, epochs=epochs, token_equalize=token_equalize, force_run=force_run)


if __name__ == '__main__':
    constants.PROMPT_DEBUG = False
    constants.PROMPT_LIBRARY_TRAINABLE.trainable = True
    constants.PROMPT_MODE.mode = 'softmax'
    constants.PROMPT_REDUCE_TYPE.reduce_type = 'prompt'
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    experiment(prefix=f'debug__', model_checkpoint=mcp,
               batch_size=32, task=('glue', 'mrpc'), gpu=1,
               encoder_max_length=None, token_equalize=False, epochs=30,
               which_model='lib',
               optimizer_param={'learning_rate': 0.3, 'weight_decay': 1E-5, 'beta_1': 0.8, 'beta_2': 0.999},
               force_run=False)

    # import numpy as np
    # first_tensor = tf.convert_to_tensor(np.arange(24).reshape((3, 2, 4)).astype('float'))
    # second_tensor = tf.convert_to_tensor(np.arange(8).reshape((1, 2, 4)).astype('float'))
    # _cosine_similarity(first_tensor, second_tensor)

    # experiment(prefix='spt_expt', model_checkpoint=mcp, batch_size=32, task=('glue', 'mrpc'),
    #            gpu=1, encoder_max_length=None, token_equalize=True, epochs=1, which_model='soft',
    #            optimizer_param={'learning_rate': 3.0, 'weight_decay': 1E-4, 'beta_1': 0.8, 'beta_2': 0.999},
    #            force_run=False)
    # constants.SEED = 73
    # hyperparameter(prefix='hp1', model_checkpoint=mcp, batch_size=32, task=('super_glue', 'wsc.fixed'),
    #                gpu=0, target_steps=20000)
