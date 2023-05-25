import torch
from transformers import T5ForConditionalGeneration

# T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("t5-base")