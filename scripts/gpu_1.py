from scripts.experiments import experiment


if __name__ == '__main__':
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    experiment(prefix='optimization_1', gpu=1, epochs=30, model_checkpoint=model_checkpoint_, max_batch_size=32,
               task=('super_glue', 'cb'))

