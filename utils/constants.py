ENCODER_MAX_LEN = 250
DECODER_MAX_LEN = 50
NUM_SOFT_TOKENS = 20

# Number of elements in each train set
COUNTS = {('glue', 'cola'): 8551, ('glue', 'mrpc'): 3668, ('glue', 'qnli'): 104743, ('glue', 'qqp'): 363846,
          ('glue', 'rte'): 2490, ('glue', 'sst2'): 67349, ('glue', 'wnli'): 635, ('glue', 'stsb'): 5749,
          ('glue', 'mnli'): 392702, ('super_glue', 'boolq'): 9427, ('super_glue', 'rte'): 2490,
          ('super_glue', 'wic'): 5428, ('super_glue', 'wsc.fixed'): 554, ('super_glue', 'multirc'): 27243,
          ('super_glue', 'cb'): 250, ('super_glue', 'copa'): 400}


def get_tasks(benchmark):
    """

    Args:
        benchmark:

    Returns:

    """

    if benchmark == 'target':
        tasks = (('super_glue', 'rte'), ('super_glue', 'multirc'), ('glue', 'mrpc'), ('glue', 'sst2'), ('glue', 'mnli'))
    elif benchmark == 'glue':
        tasks = (('glue', 'cola'), ('glue', 'mrpc'), ('glue', 'qnli'), ('glue', 'qqp'),
                 ('glue', 'rte'), ('glue', 'sst2'), ('glue', 'wnli'), ('glue', 'stsb'), ('glue', 'mnli'))
    elif benchmark == 'superglue':
        tasks = (('super_glue', 'boolq'), ('super_glue', 'rte'), ('super_glue', 'wic'), ('super_glue', 'wsc.fixed'),
                 ('super_glue', 'multirc'), ('super_glue', 'cb'), ('super_glue', 'copa'))
    else:
        raise KeyError(f'Benchmark {benchmark} is not supported')

    return tasks
