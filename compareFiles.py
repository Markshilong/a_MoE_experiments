def compare_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()
        if content1 == content2:
            print("Same.")
        else:
            print("NOT same")

# # Example usage:
# compare_files("/home/mark/Research/nvme_offload_save/zero_stage_3/float32params/rank0/456_param.tensor.swp", "/home/mark/Research/nvme_offload/zero_stage_3/float32params/rank0/456_param.tensor.swp")
# # compare_files("/home/mark/Research/a_MoE_experiments/before.txt", "/home/mark/Research/a_MoE_experiments/after.txt")
# # compare_files("/home/mark/Research/a_MoE_experiments/metaDataBefore.txt", "/home/mark/Research/a_MoE_experiments/metaDataAfter.txt")
# # diff "/home/mark/Research/nvme_offload_save/zero_stage_3/float32params/rank0/0_param.tensor.swp" "/home/mark/Research/nvme_offload/zero_stage_3/float32params/rank0/0_param.tensor.swp"
# # compare_files("/home/mark/Research/a_MoE_experiments/a.txt", "/home/mark/Research/a_MoE_experiments/b.txt")
import os

def get_file_names(directory):
    file_names = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            file_names.append(file)
    return file_names

# Specify the directory path
directory_path = '/home/mark/Research/nvme_offload_save/zero_stage_3/float32params/rank0/'

# Get the file names in the directory
files = get_file_names(directory_path)

for file in files:
    print(file)
    compare_files('/home/mark/Research/nvme_offload_save/zero_stage_3/float32params/rank0/'+file, '/home/mark/Research/nvme_offload/zero_stage_3/float32params/rank0/'+file)