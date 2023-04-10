from utils import constants
# from scripts.baseline import run_soft
# from scripts.experiments import hyperparameter
# from scripts.baseline import run_fft
from scripts.baseline import run_lib, run_spt  # , run_fft


if __name__ == '__main__':
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    constants.SEED = 42
    run_lib(model_checkpoint=mcp, batch_size=32, benchmark='super_glue',
            prefix='lib', token_equalize=False, gpu=1, force_run=False, target_steps=30000, epochs=None,
            optimizer_params={'learning_rate': 0.1, 'weight_decay': 1E-5, 'beta_1': 0.8, 'beta_2': 0.999},
            prompt_mode='weighted', prompt_reduce_type='prompt')

    # run_spt(model_checkpoint=mcp, batch_size=32, benchmark='glue', epochs=1,
    #         prefix='baseline_spt', token_equalize=False, gpu=1, force_run=False, target_steps=30000,
    #         optimizer_params={'learning_rate': 0.3, 'weight_decay': 1E-4, 'beta_1': 0.8, 'beta_2': 0.999},
    #         source_task=('glue', 'mrpc'))
