def compare_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()
        if content1 == content2:
            print("Same.")
        else:
            print("NOT same")

# Example usage:
# compare_files("/home/mark/Research/nvme_offload_save/zero_stage_3/float32params/rank0/511_param.tensor.swp", "/home/mark/Research/nvme_offload/zero_stage_3/float32params/rank0/511_param.tensor.swp")
compare_files("/home/mark/Research/a_MoE_experiments/before.txt", "/home/mark/Research/a_MoE_experiments/after.txt")

# diff "/home/mark/Research/nvme_offload_save/zero_stage_3/float32params/rank0/0_param.tensor.swp" "/home/mark/Research/nvme_offload/zero_stage_3/float32params/rank0/0_param.tensor.swp"
# compare_files("/home/mark/Research/a_MoE_experiments/a.txt", "/home/mark/Research/a_MoE_experiments/b.txt")