ENCODER_MAX_LEN = 250
DECODER_MAX_LEN = 50
NUM_SOFT_TOKENS = 20


def get_tasks(benchmark):
    """

    Args:
        benchmark:

    Returns:

    """

    if benchmark == 'target':
        tasks = (('super_glue', 'rte'), ('super_glue', 'multirc'), ('glue', 'mnli'), ('glue', 'mrpc'), ('glue', 'sst2'))
    elif benchmark == 'glue':
        tasks = (('glue', 'cola'), ('glue', 'mrpc'), ('glue', 'qnli'), ('glue', 'qqp'),
                 ('glue', 'rte'), ('glue', 'sst2'), ('glue', 'wnli'), ('glue', 'stsb'), ('glue', 'mnli'))
    elif benchmark == 'superglue':
        tasks = (('super_glue', 'boolq'), ('super_glue', 'rte'), ('super_glue', 'wic'), ('super_glue', 'wsc.fixed'),
                 ('super_glue', 'multirc'), ('super_glue', 'cb'), ('super_glue', 'copa'))
    else:
        raise KeyError(f'Benchmark {benchmark} is not supported')

    return tasks