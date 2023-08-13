#!/usr/bin/env python

# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 check_requirements.py
# torchrun --nproc_per_node=1 check_requirements.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 check_requirements.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 T0_3B_12GB.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 T0_3B_12GB.py

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch
import time
import psutil
import threading
from matplotlib import pyplot as plt
import subprocess
import sys
from datetime import datetime
import signal
from torch.cuda import nvtx
# from deepspeed.utils.debug import my_saveload_module_individually

def monitor():
    last_read_bytes = 0
    last_write_bytes = 0
    while True:
        # memory
        memory_usage = process.memory_info().vms / 1024 / 1024  # 获取当前进程的内存使用情况（以MB为单位）
        memory_usage_values.append(memory_usage)  # 记录当前内存使用情况
        
        # disk
        process_io = process.io_counters()
        # Calculate the read and write speeds
        read_speed_values.append(10 * (process_io.read_bytes - last_read_bytes) / 1024)  # KB/s
        write_speed_values.append(10 * (process_io.write_bytes - last_write_bytes) / 1024) # KB/s
        last_read_bytes = process_io.read_bytes
        last_write_bytes = process_io.write_bytes

        timestamps.append(time.time() - start_time)
        time.sleep(0.1)  # 每隔0.1秒钟获取一次内存使用情况
        if finish_flag: break
    
    

def print_params_embedding_layernorm(module):

    def print_weights_recursively(module):
        for child in module.children():
            print_weights_recursively(child)
        if module.__class__.__name__ == "Embedding":
            filename = "/home/mark/Research/a_MoE_experiments/paramsEmbedding_before_generate_skip.txt"
            with open(filename, 'a+') as f:
                f.write("-------------------------------------")
                f.write(f"Name[{module.__class__.__name__}]\n")
                for i, param in enumerate(module.parameters()):
                    f.write(f"[{i}]\n")
                    f.write(f"[param.ds_id]{param.ds_id}\n")
                    f.write(f"[param.ds_numel]{param.ds_numel}\n")
                    f.write(f"[param.ds_shape]{param.ds_shape}\n")
                    f.write(f"[param.data]{param.data}\n")
                    f.write(f"[param.data.shape]{param.data.shape}\n")
                    f.write(f"[param.ds_tensor]{param.ds_tensor}\n\n")
                # f.write(f"Name[{module.state_dict()}]\n\n")
        elif module.__class__.__name__ == "T5LayerNorm":
            filename = "/home/mark/Research/a_MoE_experiments/paramsLayerNorm_before_generate_skip.txt"
            with open(filename, 'a+') as f:
                f.write("-------------------------------------")
                f.write(f"Name[{module.__class__.__name__}]\n")
                for i, param in enumerate(module.parameters()):
                    f.write(f"[{i}]\n")
                    f.write(f"[param.ds_id]{param.ds_id}\n")
                    f.write(f"[param.ds_numel]{param.ds_numel}\n")
                    f.write(f"[param.ds_shape]{param.ds_shape}\n")
                    f.write(f"[param.data]{param.data}\n")
                    f.write(f"[param.data.shape]{param.data.shape}\n")
                    f.write(f"[param.ds_tensor]{param.ds_tensor}\n\n")

                # f.write(f"Name[{module.state_dict()}]\n\n")


    if os.path.exists("/home/mark/Research/a_MoE_experiments/paramsEmbedding_before_generate_skip.txt"):
        os.remove("/home/mark/Research/a_MoE_experiments/paramsEmbedding_before_generate_skip.txt")
        os.remove("/home/mark/Research/a_MoE_experiments/paramsLayerNorm_before_generate_skip.txt")
    print_weights_recursively(module)



# ----- monitor -----
process = psutil.Process()
pid = process.pid
print(f"------------ PID = {pid} ----------------")
    
# 记录时间和内存使用情况的列表
timestamps = []
memory_usage_values = []
read_speed_values = []
write_speed_values = []
start_time = time.time()
# 创建并启动监控线程
finish_flag = False
# monitor_thread = threading.Thread(target=monitor)
# monitor_thread.start()

# ------------------------------------------------------



time0 = time.time()
print(f"--------------- [0s] Start time -------------")

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
# deepspeed.init_distributed()

# model_name = "bigscience/T0"
model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size
# train_batch_size = 1024
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

# keeping the same format as json for consistency, except it uses lower case for true/false
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
            "nvme_path": "/home/shilonglei/OOC/nvme_offload",
            "pin_memory": True,
            "buffer_count": 6,
            "buffer_size": 1e8,
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
        "block_size": 131072,
        "queue_depth": 16,
        "thread_count": 1,
        "single_submit": True,
        "overlap_events": True
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# ('read', 'single', 'overlap', 1, 1, 16, 131072) = 2.0076069482021723

# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail

# this line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
nvtx.range_push("HfDeepSpeedConfig(ds_config)")
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
nvtx.range_pop()

nvtx.range_push("model.from_pretrained")
# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, low_cpu_mem_usage=True)
nvtx.range_pop()
time1 = time.time()
print(f"\n--------[{time1 - time0}s] model.from_pretrained DONE, interval:{time1 - time0} -------------\n")
# exit()

# initialise Deepspeed ZeRO and store only the engine object
nvtx.range_push("deepspeed.initialize")
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
nvtx.range_pop()
time2 = time.time()
print(f"\n--------[{time2 - time0}s] deepspeed.initialize DONE, interval:{time2 - time1} -------------\n")


ds_engine.module.eval()  # inference


# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
# if rank == 0:
#     text_in = "what do you think of president Obama?"
#     # text_in = "I really want to eat an apple now"
# elif rank == 1:
#     text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"
text_in = "what do you think of president Obama?"
batch_size = 1024
from copy import deepcopy
text_ins = [deepcopy(text_in) for _ in range(batch_size)]

tokenizer = AutoTokenizer.from_pretrained(model_name)
time3 = time.time()
print(f"\n--------[{time3 - time0}s] tokenizer.from_pretrained() DONE, interval:{time3 - time2} -------------\n")

# inputs_1 = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
encoded_inputs = tokenizer.batch_encode_plus(text_ins, padding=True, truncation=True, return_tensors="pt").to(device=local_rank)
input_ids = encoded_inputs["input_ids"]
# attention_mask = encoded_inputs["attention_mask"]


#from transformers.deepspeed import is_deepspeed_zero3_enabled
#print(f"Deepspeed 3 is enabled: {is_deepspeed_zero3_enabled()}")


for i in range(2):
    print(f"monitor will start in {i+1} seconds")
    time.sleep(1)


print(f"start inference in 1 seconds")
time.sleep(1)





# -------------------------INFERENCE start-------------------------
print(" !!! Start inference at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
inf_start = time.time()

iteration_num = 0

with torch.no_grad():
    nvtx.range_push("inference")
    outputs = ds_engine.module.generate(input_ids, synced_gpus=True)
    nvtx.range_pop()



inf_end = time.time()
print(" !!! End inference at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# -------------------------INFERENCE end-------------------------
print(f"\n--------------------------- inference time = {inf_end - inf_start}s --- iteration:{iteration_num}--------------------\n")



nvtx.range_push("decode")
text_out = []
for i in range(3):
    text_out.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
nvtx.range_pop()

time4 = time.time()
print(f"\n--------[{time4 - time0}s] encode + Generate + decode DONE, interval:{time4 - time3} -------------\n")

print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")

print("\n\n------------time summary ---------------")
print(f"[0s] Start time")
print(f"[{time1 - time0}s] model.from_pretrained DONE, interval:{time1 - time0}s")
print(f"[{time2 - time0}s] deepspeed.initialize DONE, interval:{time2 - time1}s")
print(f"[{time3 - time0}s] tokenizer.from_pretrained() DONE, interval:{time3 - time2}s")
print(f"!! inference time = {inf_end - inf_start}s ")
print(f"[{time4 - time0}s] encode + Generate + decode DONE, interval:{time4 - time3}s")

finish_flag = True
