import os
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
    weight_decays = [1E-5, 1E-3]
    learning_rates = [0.3]
    beta1_decays = [0.7, 0.8, 0.9]
    beta2_decays = [0.99, 0.999, 0.9999]
    do_schedulers = [False, ]

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


def hyperparameter(prefix='hyperparameter', model_checkpoint='t5-small', max_batch_size=100, task=None, epochs=30,
                   gpu=0):
    """

    Args:
        prefix:
        model_checkpoint: t5-small, t5-base
        max_batch_size: Maximum batch size
        epochs: Number of training epochs
        gpu: Which GPU to use
        task:

    Returns:
    """
    which_model = 'soft'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    model_checkpoint = model_config['model_checkpoint']
    which_model = model_config['which_model']
    logger = create_logger(checkpoint_filepath, filename=f'{prefix}-{model_checkpoint}-{which_model}-benchmark.log')
    logger.info(f'Performing {task} tuning')

    # Ensure at least 50 batches
    batch_size = {task: max_batch_size}

    # Get the batch size
    # Run one experiment and log all results
    # If it fails then carry on
    total_steps = int((constants.COUNTS[task] / batch_size[task]) * epochs)
    optimizer_experiments = _create_optimizer_experiments(total_steps)

    for optimizer_param in optimizer_experiments:
        optimizer_param_ = get_optimizer(optimizer_param)
        run_one_split(logger, model_config=model_config, optimizer_params=optimizer_param_, which_data=task,
                      batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                      debug=True, prefix=prefix, epochs=epochs)


def experiment(prefix='experiment', model_checkpoint='t5-small', max_batch_size=100, min_num_batches=50, task=None,
               epochs=None, gpu=0, optimizer_param=None, encoder_max_length=None):
    """

    Args:
        prefix:
        model_checkpoint: t5-small, t5-base
        max_batch_size: Maximum batch size
        min_num_batches: Minimum number of batches
        epochs: Number of training epochs
        gpu: Which GPU to use
        task:
        optimizer_param:
        encoder_max_length: Max length to encode the inputs

    Returns:
    """
    which_model = 'soft'
    target_steps = 30000

    default = {'learning_rate': 0.1, 'weight_decay': 1E-3}
    optimizer_param = default if optimizer_param is None else optimizer_param

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    model_config = {'model_checkpoint': model_checkpoint, 'which_model': which_model,
                    'encoder_max_length': encoder_max_length}
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    # Create a log object
    model_checkpoint = model_config['model_checkpoint']
    which_model = model_config['which_model']
    logger = create_logger(checkpoint_filepath, filename=f'{prefix}-{model_checkpoint}-{which_model}-benchmark.log')
    logger.info(f'Performing {task} tuning')

    # Ensure at least 50 batches
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task] / min_num_batches))}

    if epochs is None:
        epochs = int(target_steps/(constants.COUNTS[task]/batch_size[task]))

    # Get the batch size
    # Run one experiment and log all results
    # If it fails then carry on
    optimizer_param_ = get_optimizer(optimizer_param)
    run_one_split(logger, model_config=model_config, optimizer_params=optimizer_param_, which_data=task,
                  batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                  debug=True, prefix=prefix, epochs=epochs)


if __name__ == '__main__':
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    # hyperparameter(prefix='betas', model_checkpoint=mcp, max_batch_size=32, task=('super_glue', 'wic'), gpu=0,
    #                epochs=50)
    experiment(prefix='encoder_length', model_checkpoint=mcp, max_batch_size=32, task=('super_glue', 'wsc.fixed'),
               gpu=0, epochs=1, encoder_max_length=300,
               optimizer_param={'learning_rate': 0.3, 'weight_decay': 1E-4, 'beta_1': 0.8, 'beta_2': 0.999})
