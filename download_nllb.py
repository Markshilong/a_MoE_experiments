from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("------- loading tokenizer -------")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")

print("------- loading model -------")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")
print("------- successfuly download model -------")
print("--- exit ---")