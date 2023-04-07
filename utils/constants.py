ENCODER_MAX_LEN = 250
DECODER_MAX_LEN = 50
NUM_SOFT_TOKENS = 20
SEED = 42

# Number of elements in each train set
COUNTS = {('glue', 'cola'): 8551, ('glue', 'mrpc'): 3668, ('glue', 'qnli'): 104743, ('glue', 'qqp'): 363846,
          ('glue', 'rte'): 2490, ('glue', 'sst2'): 67349, ('glue', 'wnli'): 635, ('glue', 'stsb'): 5749,
          ('glue', 'mnli'): 392702,
          ('super_glue', 'boolq'): 9427, ('super_glue', 'rte'): 2490,
          ('super_glue', 'wic'): 5428, ('super_glue', 'wsc.fixed'): 554, ('super_glue', 'multirc'): 27243,
          ('super_glue', 'cb'): 250, ('super_glue', 'copa'): 400}

GLUE_TASKS = (('glue', 'mnli'), ('glue', 'rte'), ('glue', 'cola'), ('glue', 'mrpc'), ('glue', 'qnli'), ('glue', 'qqp'),
              ('glue', 'sst2'), ('glue', 'wnli'))
SUPERGLUE_TASKS = (('super_glue', 'cb'), ('super_glue', 'boolq'), ('super_glue', 'wic'),
                   ('super_glue', 'multirc'), ('super_glue', 'copa'),  ('super_glue', 'wsc.fixed'),)
TARGET_TASKS = (('super_glue', 'rte'), ('super_glue', 'multirc'), ('glue', 'mrpc'), ('glue', 'sst2'), ('glue', 'mnli'))


class Tasks:

    def __getitem__(self, item):
        if item.lower() == 'glue':
            return GLUE_TASKS
        elif item.lower() in ['superglue', 'super_glue']:
            return SUPERGLUE_TASKS
        elif item.lower() in ['target', 'target_tasks', 'target tasks']:
            return TARGET_TASKS
        elif item.lower() in ['source', 'source_tasks', 'source tasks']:
            return tuple([task for task in GLUE_TASKS + SUPERGLUE_TASKS if task not in TARGET_TASKS])


if __name__ == '__main__':
    print(Tasks()['glue'])
