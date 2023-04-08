from utils import constants
# from scripts.baseline import run_soft
from scripts.experiments import hyperparameter


if __name__ == '__main__':
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    # constants.SEED = 47
    # run_soft(model_checkpoint=model_checkpoint_, batch_size=32, benchmark='superglue',
    #          prefix='baseline_soft_unequal_steps',
    #          token_equalize=False, gpu=1, target_steps=30000,
    #          optimizer_params={'learning_rate': 0.3, 'weight_decay': 1E-5, 'beta_1': 0.8, 'beta_2': 0.999})
    constants.SEED = 73
    hyperparameter(prefix='hp2', model_checkpoint=mcp, batch_size=32, task=('super_glue', 'boolq'), gpu=1,
                   target_steps=20000)
