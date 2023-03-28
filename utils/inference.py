import os
from transformers import T5Tokenizer
from utils.model import TFT5ForConditionalGeneration

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


mcp = 'google/t5-base-lm-adapt'
tokenizer = T5Tokenizer.from_pretrained(mcp)
model = TFT5ForConditionalGeneration.from_pretrained(mcp)

input_s = ["What is the meaning of this?", "He is happy because he is contributing? True or False?"]
input_ids = tokenizer(input_s, padding=True).input_ids
print(input_ids)
outputs = model.generate(input_ids, max_length=10)

for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))
