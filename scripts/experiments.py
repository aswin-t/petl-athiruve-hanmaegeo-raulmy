import os
import pickle
import tensorflow as tf
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


def experiment(prefix='experiment', model_checkpoint='t5-small', max_batch_size=100, min_num_batches=50, task=None,
               epochs=30, gpu=0, lr=0.3):
    """

    Args:
        prefix:
        model_checkpoint: t5-small, t5-base
        max_batch_size: Maximum batch size
        min_num_batches: Minimum number of batches
        epochs: Number of training epochs
        gpu: Which GPU to use
        task:
        lr:

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
    batch_size = {task: min(max_batch_size, int(constants.COUNTS[task] / min_num_batches))}

    # Get the batch size
    # Run one experiment and log all results
    # If it fails then carry on
    optimizer_lrs = {task: lr}
    optimizer_params = get_optimizer(optimizer_lrs, which_data=task)
    run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                  batch_size=batch_size[task], cache_path=cache_path, checkpoint_filepath=checkpoint_filepath,
                  debug=True, prefix=prefix, epochs=epochs)


if __name__ == '__main__':
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    experiment(prefix='experiment', model_checkpoint=mcp, max_batch_size=32, min_num_batches=50, task=('glue', 'mrpc'),
               epochs=300, gpu=0)
