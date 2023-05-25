#!/usr/bin/env python

# from: https://github.com/huggingface/transformers/issues/16616
# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 nllb_ZeRO_inference.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 nllb_ZeRO_inference.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py


from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "False"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

# model_name = "bigscience/T0"
# model_name = "bigscience/T0_3B"
model_name = "facebook/nllb-moe-54b"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For indepth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# XXX: modified this script to use nvme offload so need to explain the new configs, but the key is
# to change the path to `nvme_path`

# keeping the same format as json for consistency, except it uses lower case for True/False
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/home/mark/Research/nvme_offload",
            "buffer_count": 1,
            "buffer_size": 6e8,
            "max_in_cpu": 1e9
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "aio": {
        "block_size": 262144,
        "queue_depth": 32,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# ds_config = {
#     "fp16": {
#         "enabled": "auto",
#         "loss_scale": 0,
#         "loss_scale_window": 1000,
#         "initial_scale_power": 16,
#         "hysteresis": 2,
#         "min_loss_scale": 1
#     },

#     "optimizer": {
#         "type": "AdamW",
#         "params": {
#             "lr": "auto",
#             "betas": "auto",
#             "eps": "auto",
#             "weight_decay": "auto"
#         }
#     },

#     "scheduler": {
#         "type": "WarmupLR",
#         "params": {
#             "warmup_min_lr": "auto",
#             "warmup_max_lr": "auto",
#             "warmup_num_steps": "auto"
#         }
#     },

#     "zero_optimization": {
#         "stage": 3,
#         "offload_optimizer": {
#             "device": "nvme",
#             "nvme_path": "/home/mark/Research/nvme_offload_path",
#             "pin_memory": True
#         },
#         "offload_param": {
#             "device": "nvme",
#             "nvme_path": "/home/mark/Research/nvme_offload_path",
#             "pin_memory": True
#         },
#         "overlap_comm": True,
#         "contiguous_gradients": True,
#         "sub_group_size": 1e14,
#         "reduce_bucket_size": "auto",
#         "stage3_prefetch_bucket_size": "auto",
#         "stage3_param_persistence_threshold": "auto",
#         "stage3_max_live_parameters": 1e9,
#         "stage3_max_reuse_distance": 1e9,
#         "stage3_gather_fp16_weights_on_model_save": True
#     },
#     "steps_per_print": 2000,
#     "train_batch_size": train_batch_size,
#     "train_micro_batch_size_per_gpu": 1,
#     "wall_clock_breakdown": False
# }
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, low_cpu_mem_usage=True)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "what do you think of president Obama?"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"


tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt",padding = True).to(device=local_rank)
#from transformers.deepspeed import is_deepspeed_zero3_enabled
#print(f"Deepspeed 3 is enabled: {is_deepspeed_zero3_enabled()}")
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")