import torch
from transformers import T5ForConditionalGeneration, PreTrainedModel
from utils import count

def func():
    global count
    count = count + 1

func()
print(count)