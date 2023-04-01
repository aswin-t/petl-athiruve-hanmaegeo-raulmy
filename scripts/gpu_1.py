from scripts.baseline import run_soft


if __name__ == '__main__':
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(model_checkpoint=model_checkpoint_, batch_size=32, benchmark='glue', prefix='baseline_soft_equal',
             token_equalize=True, gpu=1)
