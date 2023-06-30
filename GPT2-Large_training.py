from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot")

config = AutoConfig.from_pretrained("codeparrot/codeparrot")



dataset = load_dataset("lvwerra/codeparrot-clean", split="train", streaming=True)