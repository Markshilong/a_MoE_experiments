import time
import psutil
import subprocess
import sys



# /home/mark/anaconda3/envs/deepspeed/bin/python

process = psutil.Process()
pid = process.pid
# strace_command = "sudo strace -o test_strace.txt -e trace=%desc -f -t -p "+str(pid)
# monitor_process = subprocess.Popen(strace_command, shell=True)

print(f"----------- PID = {pid} ----------------")
while True:
    with open("test.txt", "w") as f:
        f.write("asdfasdfasdfasfasadfad")
    print("just wrote 1")
    with open("test_2.txt", "w") as f:
        f.write("asdfasdfasdfasfasadfad")
    print("just wrote 2")
    time.sleep(1)