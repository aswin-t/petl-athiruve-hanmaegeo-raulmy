import os
import pickle
import tensorflow as tf
from transformers import AutoTokenizer
from transformers.models.t5 import TFT5ForConditionalGeneration
from utils import constants
from utils.constants import Tasks
from utils.data import PrepDataset
from utils.log import create_logger
from utils.train import run_one_split, run_lr_split, get_optimizer

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


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
    which_model = 'fft'

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
    # task = ('super_glue', 'multirc')
    task = ('super_glue', 'boolq')
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task]/min_num_batches))}

    # for lr in [1E-7, 1E-6, 3E-6, 1E-5, 1E-4, 1E-3, 1E-3, 1E-1, 1, 10]:
    for lr in [1E-2, ]:
        # Get the batch size
        # Run one experiment and log all results
        # If it fails then carry on
        optimizer_lrs = {task: lr}
        optimizer_params = get_optimizer(optimizer_lrs, which_data=task)
        run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                      batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                      debug=True, prefix=prefix)


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
