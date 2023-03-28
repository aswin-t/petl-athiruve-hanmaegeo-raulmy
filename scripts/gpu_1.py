from scripts.baseline import run_soft


if __name__ == '__main__':
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    run_soft(benchmark='glue', gpu=1, epochs=200, model_checkpoint=model_checkpoint_, max_batch_size=32)
