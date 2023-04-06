from utils import constants
from scripts.baseline import run_soft


if __name__ == '__main__':
    constants.SEED = 47
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(model_checkpoint=model_checkpoint_, batch_size=32, benchmark='superglue',
             prefix='baseline_soft_unequal_lr',
             token_equalize=False, gpu=1, target_steps=15000,
             optimizer_params={'learning_rate': 0.1, 'weight_decay': 1E-4, 'beta_1': 0.8, 'beta_2': 0.999})
