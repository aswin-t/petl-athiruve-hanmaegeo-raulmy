from utils import constants
from scripts.baseline import run_spt


if __name__ == '__main__':
    mcp = 'google/t5-base-lm-adapt'.replace('/', '_-_')
    constants.SEED = 42
    # run_lib(model_checkpoint=mcp, batch_size=32, benchmark='super_glue_bugs', prefix='lib_bug_', token_equalize=False,
    #         gpu=1, force_run=False, target_steps=30000, epochs=None,
    #         optimizer_params={'learning_rate': 0.3, 'weight_decay': 1E-5, 'beta_1': 0.8, 'beta_2': 0.999},
    #         prompt_mode='weighted', prompt_reduce_type='token', prompt_library_trainable=True)
    source_task = ('glue', 'wnli')
    prefix = f'spt-{source_task[0]}-{source_task[1]}'
    run_spt(model_checkpoint=mcp, batch_size=32, benchmark='super_glue',
            prefix=prefix, token_equalize=False, gpu=0, force_run=False, target_steps=30000,
            optimizer_params={'learning_rate': 0.1, 'weight_decay': 1E-4, 'beta_1': 0.8, 'beta_2': 0.999},
            source_task=source_task)
