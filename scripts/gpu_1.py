from scripts.baseline import run_soft


if __name__ == '__main__':
    run_soft(benchmark='super_glue', gpu=1, epochs=30)
