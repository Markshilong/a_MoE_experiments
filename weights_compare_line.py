import os.path

file1 = '/home/mark/Research/a_MoE_experiments/weights_skip_1.txt'  # First file name
file2 = '/home/mark/Research/a_MoE_experiments/weights_ori_1.txt'  # Second file name
file1_basename = os.path.basename(file1)
file2_basename = os.path.basename(file2)
# Open the files for reading
with open(file1, 'r') as f1, open(file2, 'r') as f2:
    # Read the lines from each file
    lines1 = f1.readlines()
    lines2 = f2.readlines()

# Compare the lines and find the differences
diff_lines_start = []
i_lastPrint = -10
for i, (line1, line2) in enumerate(zip(lines1, lines2)):
    if i > 10000: break


    if line1 != line2:
        if i == i_lastPrint + 1:
            i_lastPrint = i
            continue
        print(f"----------line {i+1}----------:")
        print(f"[{file1_basename}]:{line1.strip()}")
        print(f"[{file2_basename}]:{line2.strip()}")
        print()
        i_lastPrint = i
        diff_lines_start.append(i+1)

if i_lastPrint == -10:
    print(f"{file1_basename} and {file2_basename} is All Same")
else:
    print(f"diff_lines_start = {diff_lines_start}")