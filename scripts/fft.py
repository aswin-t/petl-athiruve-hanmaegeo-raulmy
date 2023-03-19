import os
from utils.train import run_benchmark
from keras.optimizers.optimizer_experimental.adamw import AdamW


def run_model(model_config: dict = None, optimizer_params=None, debug: bool = False, prefix=''):
    """

    Args:
         model_config: Dictionary:
                'model_checkpoint': <t5-small, t5-base>
                'which_model': 'fft' -> full fine-tuning of all parameters.
                               'soft' -> soft prompt is tuned
                               'library' -> library of soft prompts
                'epochs': <Optional>
        optimizer_params: {'optimizer': <Object of optimizer class or None to use default>, 'tag': <text description>}
        debug: if True then eager model of evaluation is run, else graph mode
        prefix: Prefix to add to the model names
    Returns:

    """

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    checkpoint_filepath = os.path.join(os.path.dirname(__file__), "../checkpoints")
    optimizer_default_params = {'algo': 'adam', 'params': {'learning_rate': 0.0001}}
    optimizer_params = optimizer_default_params if optimizer_params is None else optimizer_params
    batch_size = 100

    # Benchmark can be given as this tuple of atsks or a benchmark name such as 'glue' or 'super_glue'
    tasks = (('super_glue', 'rte'), ('super_glue', 'multirc'), ('glue', 'mnli'), ('glue', 'mrpc'), ('glue', 'sst2'))

    #  Run the superglue benchmnark
    run_benchmark(model_config=model_config, optimizer_params=optimizer_params, batch_size=batch_size,
                  cache_path=cache_path, checkpoint_filepath=checkpoint_filepath, debug=debug, benchmark=tasks,
                  one_task=None, prefix=prefix)


if __name__ == '__main__':
    model_checkpoint = 't5-small'
    which_model = 'fft'
    learning_rate = 0.001
    mc = {'model_checkpoint': model_checkpoint, 'which_model': which_model, 'epochs': 50}
    op = {'tag': f'adamw-learning_rate-{learning_rate:.6f}',
          'optimizer': AdamW(learning_rate=learning_rate)}
    run_model(model_config=mc, optimizer_params=op, debug=False, prefix='athiruve')
