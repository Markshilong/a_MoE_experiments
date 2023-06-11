from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np

nvmeOffload_logFile = "/home/mark/Research/a_MoE_experiments/monitors/gpu_trace/nvidiasmi_nvmeOffload.txt"
cpuOffload_logFile = "/home/mark/Research/a_MoE_experiments/monitors/gpu_trace/nvidiasmi_cpuOffload.txt"
noOffload_logFile = "/home/mark/Research/a_MoE_experiments/monitors/gpu_trace/nvidiasmi_noOffload.txt"


def prepare_data():
    # HAS start and end
    # cpu offload
    # 17:38:20
    # 17:38:26
    time_start = datetime.strptime("17:38:20", time_format)
    time_end = datetime.strptime("17:38:26", time_format)


    duration = time_end - time_start
    SecondsNum = duration.seconds

    timestamps = []
    GPU_utilization = []
    GPU_memory = []


    last_time_obj = time_start
    last_time_str = ""
    for line in lines:
        # Extracting the desired information
        time_index = line.index(':') - 2
        time_str = line[time_index:time_index + 8]
        time_obj = datetime.strptime(time_str, time_format)

        if (time_obj >= time_start and time_obj <= time_end):
            if (time_str == last_time_str):
                time_obj = last_time_obj + timedelta(milliseconds=100)
            last_time_obj = time_obj
            last_time_str = time_str
            timestamps.append(time_obj)

            percent_index = line.find('%')
            percent = line[percent_index - 2:percent_index + 1]

            memory_index_1 = line.find('MiB', percent_index - 10)
            memory_index = line.find('MiB', memory_index_1 + 1)
            memory = line[memory_index - 5:memory_index + 3]

            GPU_utilization.append(int(percent[:-1]))
            GPU_memory.append((float(memory[:-3])/6144)*100)

            # print(f"Time: {time_str}, GPU Utilization: {percent}, Memory Usage: {memory}")

with open(nvmeOffload_logFile, "r") as f:
    content = f.read()

time_format = '%H:%M:%S'
lines = content.split("&")
if(len(lines[-1]) < 10): lines.pop(-1)


# Start inference at 2023-06-10 15:39:36
# End inference at 2023-06-10 15:40:03


# --------------- no start and end --------------------------
# the first item time is start time
line = lines[0]
start_time_index = line.index(':') - 2
start_time_str = line[start_time_index:start_time_index + 8]
start_time_obj = datetime.strptime(start_time_str, time_format)


timestamps = []
GPU_utilization = []
GPU_memory = []


last_time_obj = start_time_obj
last_time_str = ""
for line in lines:
    # Extracting the desired information
    time_index = line.index(':') - 2
    time_str = line[time_index:time_index + 8]
    time_obj = datetime.strptime(time_str, time_format)

    if (time_str == last_time_str):
        time_obj = last_time_obj + timedelta(milliseconds=100)
    last_time_obj = time_obj
    last_time_str = time_str
    

    percent_index = line.find('%')
    percent = line[percent_index - 2:percent_index + 1]

    memory_index_1 = line.find('MiB', percent_index - 10)
    memory_index = line.find('MiB', memory_index_1 + 1)
    memory = line[memory_index - 5:memory_index + 3]

    # collect data
    timestamps.append(time_obj)
    GPU_utilization.append(int(percent[:-1]))
    GPU_memory.append((float(memory[:-3])/6144)*100)

    print(f"Time: {time_str}, GPU Utilization: {percent}, Memory Usage: {memory}")


# -----------------------------------------------------------

# print(timestamps)


# Create Figure
fig1 = plt.figure()
x1 = timestamps

# ax1
ax1 = fig1.add_subplot(111)
ax1.plot(x1, GPU_utilization, 'b-')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('GPU utilization(%)')
ax1.set_ylim(top=100)

# ax2
ax2 = ax1.twinx()
ax2.plot(x1, GPU_memory, 'red')
ax2.set_ylabel('GPU memory usage(%)')
ax2.set_ylim(top=100)



plt.show()
