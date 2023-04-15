# petl-athiruve-hanmaegeo-raulmy
Exploring a generalized solution for soft prompt transfer

All experiment results and model checkpoints are saved in the mycheckpoints folder. The log file stores the experiment results

# Full fine tuning baseline
- Git clone this repository

- Navigate to the notebooks folder and open optimization.ipynb
- Choose your benchmark 'glue' or 'super_glue'
- Run the notebook, it will create a file in the mycheckpoints/optimize folder witha  pickle of learning rates
- Now open the fft.ipynb notebook and set the same benchmark as above and run it

# Vanilla soft prompt tuning
- Git clone this repository
- Navigate to the notebooks folder and open soft.ipynb
- Choose your benchmark 'glue' or 'super_glue'
- Run the notebook
- The soft prompts 

# Soft prompt transfer
- Git clone this repository
- Navigate to the notebooks folder and open spt.ipynb
- Note that soft prompt must have already been run and the mycheckpoints/sof_prompts folder is already created
- choose your source task 
- Choose your benchmark 'glue' or 'super_glue'
- Run the notebook

# Library prompt transfer
- Git clone this repository
- Navigate to the notebooks folder and open libprompt.ipynb
- Note that soft prompt must have already been run and the mycheckpoints/sof_prompts folder is already created
- Choose your benchmark 'glue' or 'super_glue'
- Run the notebook