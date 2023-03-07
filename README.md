# petl-athiruve-hanmaegeo-raulmy
Exploring a generalized solution for soft prompt transfer

## Getting started

### If setting up on your local machine 
1. Follow instructions to install transformers from Hugging Face (https://huggingface.co/docs/transformers/installation) 
   and follow instruction listed below
2. Clone this repository to your local machine
3. Locate on you machine where the
4. From the 'utils' folder of this repository copy ('transformer.model.t5__init__.py')
5. Find the '__init__.py' file from the folder transfomers -> model -> t5 and rename it to something else. Perhaps '__init__backup.py'
6. Copy 'transformer.model.t5__init__.py' into this folder and rename it to '__init__.py'

There are some Classes from T5 models that are not made available by default. This modification to the '__init__.py' 
exposes those subclasses

