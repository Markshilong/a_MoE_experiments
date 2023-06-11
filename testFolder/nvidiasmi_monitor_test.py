import os
import psutil
import subprocess
from datetime import datetime
import time
import signal
# from deepspeed.utils.debug import my_saveload_module_individually
current_directory = os.path.dirname(os.path.abspath(__file__))
    
nvidia_monitor_enabled = True #
nvidia_command = "/home/mark/Research/a_MoE_experiments/monitors/nvidia-smi_prof.sh " + current_directory + "/nvidiasmi_test.txt"


# ----- monitor -----
process = psutil.Process()
pid = process.pid
print(f"------------ PID = {pid} ----------------")

if (nvidia_monitor_enabled):
    nvidia_process = subprocess.Popen(nvidia_command, shell=True)
    nvidia_parent_process = psutil.Process(nvidia_process.pid)

for i in range(5):
    print(f"now {i}")
    time.sleep(1)


if (nvidia_monitor_enabled):
    for child in nvidia_parent_process.children(recursive=True):
        os.kill(child.pid, signal.SIGINT)
        print(f"just killed {child.pid}")


