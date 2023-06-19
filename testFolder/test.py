import time
import psutil
import subprocess
import sys
import os
import signal
import torch

import copy

# Input sentence
text_in = "what do you think of president Obama?"

# Number of replicas
num_replicas = 3

# Create a list of replicated sentences using deepcopy
text_ins = [copy.deepcopy(text_in) for _ in range(num_replicas)]

# Print the replicated sentences
print(text_ins)
text_ins[1] = "adfadsf"

print(text_ins)
# process = psutil.Process()
# pid = process.pid
# strace_command = "sudo strace -o strace_all_test.txt -f -t -p " # +str(pid)
# blktrace_command = "sudo blktrace -d /dev/nvme1n1p4 -o - | blkparse -i - > blkparse_test.txt"
# blktrace_command1 = "sudo blktrace -d /dev/nvme1n1p4 -o blktrace_test"
# blktrace_command2 = "sudo blkparse -i blktrace_test > blkparse_test.txt"
# # strace_command = "sudo strace -o test_strace.txt -e trace=%desc -f -t -p "+str(pid)
# # monitor_process = subprocess.Popen(strace_command, shell=True)
# # sudo blktrace -d /dev/nvme1n1p4 -o - | blkparse -i -
# # sudo blktrace -d /dev/nvme1n1p4 -o blktrace_test
# # sudo blkparse -i tracefile.blktrace -d processname > result.txt

# # 259,0    5       11     1.001611627 11492  A   W 805537640 + 8 <- (259,4) 152246120

# # data = b'0'*16  # 16 bytes
# # print(f"----------- PID = {pid} ----------------")
# # monitor_process = subprocess.Popen(blktrace_command, shell=True)
# # print(f"monitor_process pid = {monitor_process.pid}")
# # parent_process = psutil.Process(monitor_process.pid)

# # time.sleep(1)

# # for i in range(4):
# #     with open("test.txt", "wb") as f:
# #         f.write(data)
# #     print("just wrote 1")
# #     time.sleep(1)

# # for child in parent_process.children():
# #     print(f'send SIGINT to {child.pid}')
# #     result = subprocess.run(["kill", "-INT", str(child.pid)], capture_output=True, text=True)

# duration = 1 # run for 3 seconds
# count = 0

# while True:
#     print("Executing once")
#     start_time = time.time()
#     while (time.time() - start_time < duration):
#         count += 1
#     time.sleep(1)