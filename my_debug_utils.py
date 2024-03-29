# integrate all my debug utils
# -------- usage --------
# import sys
# sys.path.append('/shared_ssd_storage/shilonglei/OOC/a_MoE_experiments/my_debug_utils.py')
# from my_debug_utils import ...

# useful command
# split -l 10000 output.txt output_ -d --additional-suffix=.txt --suffix-length=3
# --- split blktrace ---
# split -l 50000 /home/mark/Research/a_MoE_experiments/blktrace_inference_ori.txt /home/mark/Research/a_MoE_experiments/inference_blktrace_strace/blktrace_inference_ori/blkparse_inference_ori_ -d --additional-suffix=.txt --suffix-length=3


# my --------------------------------------
import os
import torch
from datetime import datetime, timedelta


# ------------------- settings, should not be changed after running ------------------------- 
my_skip_1_enabled = False # True to skip initialization stage 1 (init weights and write to NVMe .swp)
# True:
# 1. my_version = True
my_skip_2_enabled = False # True to skip initialization stage 2 (module._load_from_state_dict(*args))
# True:
# 1. skip 'module._load_from_state_dict(*args)'
# 2. activate 'my_saveload_module_individually'

# ---- strace ----
strace_monitor_enabled = False # True to run strace command
# strace_command = "sudo strace -o strace_desc_ori_2.txt -e trace=%desc -f -t -p " # +str(pid)
strace_command = "sudo strace -o strace_all_inference_ori.txt -f -t -p " # +str(pid)


# ---- blktrace -----
blktrace_monitor_enabled = False # True to run blktrace command
# blktrace_command = "sudo blktrace -d /dev/nvme1n1p4 -o - | blkparse -i - > blktrace_inference_ori.txt"
blktrace_command = "sudo blktrace -d /dev/nvme1n1p4 -o blktrace_inference_ori"

forward_prehook_time_output = True

# ---- sar -----
sar_monitor_enabled = False #
sar_command = "sar -u 1 120 > sar_allCPU_inference_ori.txt"  # all cpus, per 1 second
sar_command = "sar -P ALL 1 120 > sar_seperateCPUs_inference_ori.txt"  # all cpus, per 1 second

# ---- nvidia-smi ----
nvidia_monitor_enabled = False #
nvidia_command = "/home/shilonglei/OOC/a_MoE_experiments/monitors/nvidia-smi_prof.sh /home/shilonglei/OOC/a_MoE_experiments/monitors/gpu_trace/nvidiasmi_noOffload.txt"


# ---- inference duration ---
inference_duration_enabled = False
duration = timedelta(minutes=1)
# --------------------------------------------------------------------------------------------

# ----
countt = 0
module_index = 0

# --------------------
lsl_zero_stage = 1

def my_saveload_module_individually(current_submodule, save_or_load, print=True):
    # save/load T5LayerNorm and Embedding weights
    global countt
    norm_save_path = "/home/mark/Research/save_load_path/T5LayerNorm/"
    embedding_save_path = "/home/mark/Research/save_load_path/Embeddings/"
    if current_submodule.__class__.__name__ == "Embedding":
        save_path = embedding_save_path + "Embedding_"+str(countt)+".pth"
        if save_or_load == 'save':
            torch.save(current_submodule.state_dict(), save_path)
            if print: print(f"[{current_submodule.__class__.__name__}] individually saved.")
        elif save_or_load == 'load':
            total_params = sum(p.numel() for p in current_submodule.parameters())
            if total_params == 1024:
                if os.path.exists(save_path):
                    current_submodule.load_state_dict(torch.load(save_path))
                    if print: print(f"[{current_submodule.__class__.__name__}] individually loaded. Module total_params={total_params}")
        
    elif current_submodule.__class__.__name__ == "T5LayerNorm":
        save_path = norm_save_path + "T5LayerNorm_"+str(countt)+".pth"
        if save_or_load == 'save':
            torch.save(current_submodule.state_dict(), save_path)
            if print: print(f"[{current_submodule.__class__.__name__}] individually saved.")
        elif save_or_load == 'load':
            total_params = sum(p.numel() for p in current_submodule.parameters())
            if os.path.exists(save_path):
                current_submodule.load_state_dict(torch.load(save_path))
                if print: print(f"[{current_submodule.__class__.__name__}] individually loaded. Module total_params={total_params}")
    countt = countt + 1

def my_print_params_info(printFileName, module_class_name, current_submodule):
    global module_index
    save_folder = '/home/mark/Research/a_MoE_experiments/'
    printFilePath = save_folder + printFileName
    if module_index == 0 and os.path.exists(printFilePath):
        os.remove(printFilePath)
    with open(printFilePath, 'a+') as f:
        # file_size = os.path.getsize(filename)
        if module_index < 3320 and current_submodule.__class__.__name__ == module_class_name:
            f.write("-------------------------------------")
            f.write(f"Name[{current_submodule.__class__.__name__}][{module_index}]\n")
            for i, param in enumerate(current_submodule.parameters()):
                f.write(f"[{i}]\n")
                f.write(f"[param.ds_id]{param.ds_id}\n")
                f.write(f"[param.ds_numel]{param.ds_numel}\n")
                f.write(f"[param.ds_shape]{param.ds_shape}\n")
                f.write(f"[param.data]{param.data}\n")
                f.write(f"[param.data.shape]{param.data.shape}\n")
                f.write(f"[param.ds_tensor]{param.ds_tensor}\n\n")
            # f.write(f"Name[{current_submodule.state_dict()}]\n\n")

# --------------------------------------------
