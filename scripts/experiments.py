import os
from utils.log import create_logger
from utils.train import run_one_split


def run_one():
    batch_size = 10
    debug = False
    task = ('super_glue', 'boolq')
    prefix = 'experiment'
    # model_config = {'model_checkpoint': 't5-small', 'which_model': 'soft', 'epochs': 0,
    #                 'prompt_transfer': {'model_checkpoint': 't5-small', 'which_data': ('super_glue', 'boolq')}}
    model_config = {'model_checkpoint': 't5-small', 'which_model': 'soft', 'epochs': 1}
    optimizer_params = {'algo': 'adam', 'params': {'learning_rate': 0.0001}}

    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints")

    # Create a log object
    logger = create_logger(output_path, filename=f'experiments.log')

    # Run one experiment and log all results
    # If it fails then carry on
    run_one_split(logger, model_config=model_config, optimizer_params=optimizer_params, which_data=task,
                  batch_size=batch_size, cache_path=cache_path, checkpoint_filepath=output_path, debug=debug,
                  prefix=prefix)


if __name__ == '__main__':
    run_one()
