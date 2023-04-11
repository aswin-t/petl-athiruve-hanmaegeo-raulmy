ENCODER_MAX_LEN = 250
DECODER_MAX_LEN = 50
NUM_SOFT_TOKENS = 20
NUM_lIBRARY_PROMPTS = 8
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
SUPERGLUE_TASKS = (('super_glue', 'boolq'), ('super_glue', 'multirc'), ('super_glue', 'wic'), ('super_glue', 'copa'),
                   ('super_glue', 'wsc.fixed'), ('super_glue', 'cb'), )
TARGET_TASKS = (('super_glue', 'rte'), ('super_glue', 'multirc'), ('glue', 'mrpc'), ('glue', 'sst2'), ('glue', 'mnli'))


class PromptMode:
    def __init__(self):
        self._mode = 'softmax'
        self.allowed_modes = ['softmax', 'weighted']

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, item):
        if item not in self.allowed_modes:
            raise ValueError(f'mode can only be one of {self.allowed_modes}')
        self._mode = item


class PromptReduceType:
    def __init__(self):
        self._reduce_type = 'prompt'
        self.allowed_reduce_types = ['prompt', 'token']

    @property
    def reduce_type(self):
        return self._reduce_type

    @reduce_type.setter
    def reduce_type(self, item):
        if item not in self.allowed_reduce_types:
            raise ValueError(f'Reduce types can only be one of {self.allowed_reduce_types}')
        self._reduce_type = item


class PromptLibraryTrainable:
    def __init__(self):
        self._trainable = False

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, item):
        if not isinstance(item, bool):
            raise ValueError(f'trainable is a boolean property and can only be set to True or False')
        self._trainable = item


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


PROMPT_MODE = PromptMode()
PROMPT_REDUCE_TYPE = PromptReduceType()
PROMPT_LIBRARY_TRAINABLE = PromptLibraryTrainable()
PROMPT_DEBUG = False


if __name__ == '__main__':
    # print(Tasks()['glue'])
    PROMPT_MODE.mode = 'weight'
    print(PROMPT_MODE.mode)

