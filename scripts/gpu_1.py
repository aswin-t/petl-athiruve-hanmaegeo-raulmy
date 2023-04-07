from utils import constants
from scripts.baseline import run_soft


if __name__ == '__main__':
    constants.SEED = 47
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(model_checkpoint=model_checkpoint_, batch_size=32, benchmark='superglue',
             prefix='baseline_soft_unequal_lowlr',
             token_equalize=False, gpu=1, target_steps=15000,
             optimizer_params={'learning_rate': 0.01, 'weight_decay': 0.004, 'beta_1': 0.9, 'beta_2': 0.999})
