from matplotlib import pyplot as plt
import numpy as np

blktrace_path = "/home/mark/Research/a_MoE_experiments/inference_strace/blktrace_inference_ori/blkparse_inference_ori.txt"
file_path = "/home/mark/Research/a_MoE_experiments/inference_strace/blktrace_inference_ori/blkparse_inference_ori_015.txt"


with open(file_path, 'r') as f:
    content = f.read()
lines = content.split('\n')  # Split the data into lines
if (len(lines[-1]) == 0): lines.pop()

timestamps = []
io_stages = []
offsets = []
# sizes = []

last_offset = 0
total_lines_num = 2136394
for line_index,line in enumerate(lines, start=1):
    if (line_index > (2136394/5)): break
    columns = line.split()  # Split each line into columns
    if(len(columns)<3): break

    if (columns[7][0]=='['):
        columns[7] = last_offset
    
    
    # add into list
    timestamps.append(float(columns[3]))
    io_stages.append(columns[5])
    offsets.append(int(columns[7]))

    last_offset = columns[7]

# --- Draw line prepare ---
not_found_A_timestamps = []
not_found_A_offsets = []
start_timestamps = []
end_timestamps = []
singleRead_offsets = []
for index, stage_item in enumerate(io_stages):
    if (stage_item == 'A'):
        # find the timestamp for this C
        duration_step = 1
        while True:
            C_index = index + duration_step
            if (C_index >= 49999):
                print(f"END NOT FOUND. A's index is {index}, C's index is {C_index}")
                not_found_A_timestamps.append(timestamps[index])
                not_found_A_offsets.append(offsets[index])
                break

            if ( (io_stages[C_index] == 'C'
                 or io_stages[C_index] == 'M'
                 or io_stages[C_index] == 'F')
                 and (offsets[C_index] == offsets[index])):
                
                # log the timestamp for this A
                start_timestamps.append(timestamps[index])
                singleRead_offsets.append(offsets[index])
                end_timestamps.append(timestamps[C_index])
                print(f"just fix one. A's index is {index}, C's index is {C_index}")
                break
            duration_step += 1



# ---- prepare points colors ----
rainbow_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
# {'P', 'A', 'I', 'C', 'M', 'D', 'G', 'UT', 'Q'}
# A -> Q -> G -> M -> I -> P -> D -> C   UT
# set color for different io_stages
colors = []
for io_stage in io_stages:
    if (io_stage == 'A'):
        colors.append('red')
    elif (io_stage == 'Q'):
        colors.append('orange')
    elif (io_stage == 'G'):
        colors.append('yellow')
    elif (io_stage == 'M'):
        colors.append('green')
    elif (io_stage == 'I'):
        colors.append('blue')
    elif (io_stage == 'P'):
        colors.append('indigo')
    elif (io_stage == 'D'):
        colors.append('purple')
    elif (io_stage == 'C'):
        colors.append('black')
    else:
        colors.append('saddlebrown')
# ----------------------------


# Create Figure 1
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
x1 = timestamps
y1 = offsets
# ax1.scatter(x1, y1, c=colors, s=15, alpha=0.7, edgecolors='black')
# ax1.scatter(x1, y1, c=colors, s=15, alpha=0.9)
for index, offset in enumerate(singleRead_offsets):
    ax1.plot([start_timestamps[index], end_timestamps[index]], [offset, offset], 'b-')
ax1.scatter(not_found_A_timestamps, not_found_A_offsets, c='red', s=15, alpha=0.9)

ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('read offset')
ax1.set_title('Read Distribution')
plt.show()
