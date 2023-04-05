from utils import constants
from scripts.baseline import run_soft


if __name__ == '__main__':
    constants.SEED = 43
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(model_checkpoint=model_checkpoint_, batch_size=32, benchmark='glue', prefix='baseline_soft_equal_2',
             token_equalize=True, gpu=1)
