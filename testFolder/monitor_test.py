# import subprocess
import psutil
import time
import ctypes
from memory_profiler import profile
import threading
from matplotlib import pyplot as plt
import pickle


# subprocess.Popen(["python", "mem_profile.py", f"{pid}"])
def monitor():
    process = psutil.Process()
    pid = process.pid
    print(f"------------ Monitor - PID = {pid} ----------------")

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


# @profile
def func():
    for i in range(5):
        print(f"loop {i}")
        buffer_size = 1024 * 1024 * 10  # Size of the buffer in bytes
        buffer = ctypes.create_string_buffer(buffer_size)
        with open('/home/mark/Research/a_MoE_experiments/test.txt', 'r') as f:
            content = f.read()

        time.sleep(0.5)
        del buffer
        with open('disk_rw.txt', 'a+') as f:
            for j in range(100):
                f.write(f"[{j}]TESTasldjfalsdfasdffffffsdfasdfasdfasdfasdfsfsfsafsdfsafs")
        time.sleep(0.5)
        
        # Do something with the allocated memory (e.g., process the data)
        # ...

        # Print the current memory usage


 # ----------------------------------------------------------------

# 记录时间和内存使用情况的列表
timestamps = []
memory_usage_values = []
read_speed_values = []
write_speed_values = []
start_time = time.time()
# 创建并启动监控线程
finish_flag = False
monitor_thread = threading.Thread(target=monitor)
monitor_thread.start()

func()

finish_flag = True


# with open('/home/mark/Research/a_MoE_experiments/rename/timestamps', 'wb')

# plt.plot(timestamps, memory_usage_values)
# plt.xlabel('Time')
# plt.ylabel('Memory Usage (MB)')
# plt.title('Memory Usage Over Time')
# plt.show()

# Create Figure 1
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
x1 = timestamps
y1 = read_speed_values
ax1.plot(x1, y1)
ax1.set_title('Read disk')

# Create Figure 2
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
x2 = timestamps
y2 = write_speed_values
ax2.plot(x2, y2)
ax2.set_title('Write disk')

# Create Figure 3
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
x3 = timestamps
y3 = memory_usage_values
ax3.plot(x3, y3)
ax3.set_title('memory usage')

# Show the figures
plt.show()