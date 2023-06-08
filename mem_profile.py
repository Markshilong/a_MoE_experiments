from psutil import Process, NoSuchProcess
import matplotlib.pyplot as plt
import time
import sys

pid = int(sys.argv[1])

print("-------------- Mem_profile.py START --------------")
print("PID:", pid)

virt_data = []
timestamps = []
start_time = time.time()

while True:
    try:
        process = Process(pid)
        virt_memory = (process.memory_info().vms) / (1024 * 1024)
        virt_data.append(virt_memory)
        timestamps.append(time.time() - start_time)
        print("mem_profile.py -> logged")
        # Sleep for a certain interval (e.g., 1 second)
        time.sleep(1)
    except NoSuchProcess:
        break

plt.plot(timestamps, virt_data)
plt.xlabel('Time (seconds)')
plt.ylabel('Virtual Memory Size (MB)')
plt.title('Virtual Memory Size Over Time')
plt.show()
