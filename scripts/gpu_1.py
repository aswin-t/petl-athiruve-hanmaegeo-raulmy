from scripts.experiments import experiment


if __name__ == '__main__':
    model_checkpoint_ = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    experiment(prefix='gpu_1_2', gpu=1, epochs=1, model_checkpoint=model_checkpoint_, max_batch_size=32,
               task=('glue', 'mrpc'), lr=0.3)
