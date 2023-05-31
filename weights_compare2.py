import os.path

file1 = '/home/mark/Research/a_MoE_experiments/weights_skip_1.txt'  # First file name
file2 = '/home/mark/Research/a_MoE_experiments/weights_skip_2.txt'  # Second file name
file1_basename = os.path.basename(file1)
file2_basename = os.path.basename(file2)

with open(file1, 'r') as file:
    content1 = file.read()
    string_list1 = content1.split('-------------------------------------')

with open(file2, 'r') as file:
    content2 = file.read()
    string_list2 = content2.split('-------------------------------------')

is_same = True

T5LayerNorm_diff_index = []
Embedding_diff_index = []
for i, string in enumerate(string_list1):
    if i > 2000: break
    start_index = string.find('[') + 1  # Find the index of the first '[' and add 1
    end_index = string.find(']')  # Find the index of the first ']'
    moduleName = string[start_index:end_index]

    if string_list1[i] != string_list2[i]:
        is_same = False
        # print(f"--------------------------- index {i} -----------------------")
        if moduleName == "T5LayerNorm":
            T5LayerNorm_diff_index.append(i)
        elif moduleName == "Embedding":
            Embedding_diff_index.append(i)
        else:
            print("!!!!!!!!!!!")
        # print(f"---[{file1_basename}]---\n{string_list1[i]}")
        # print(f"---[{file2_basename}]---\n{string_list2[i]}")
if is_same:
    print(f"{file1_basename} and {file2_basename} is All Same")
    exit()
print(f"T5LayerNorm_diff_index = {T5LayerNorm_diff_index}")
print(f"Embedding_diff_index = {Embedding_diff_index}")

# print all T5LayerNorm index
T5LayerNorm_index = []
for i, string in enumerate(string_list1):
    if i > 2000: break
    start_index = string.find('[') + 1  # Find the index of the first '[' and add 1
    end_index = string.find(']')  # Find the index of the first ']'
    moduleName = string[start_index:end_index]
    if moduleName == "T5LayerNorm":
        T5LayerNorm_index.append(i)
print(f"T5LayerNorm_index = {T5LayerNorm_index}")
# print all Embedding index
Embedding_index = []
for i, string in enumerate(string_list1):
    if i > 2000: break
    start_index = string.find('[') + 1  # Find the index of the first '[' and add 1
    end_index = string.find(']')  # Find the index of the first ']'
    moduleName = string[start_index:end_index]
    if moduleName == "Embedding":
        Embedding_index.append(i)
print(f"Embedding_index = {Embedding_index}")


if T5LayerNorm_index == T5LayerNorm_diff_index:
    print("all T5LayerNorm are different")
else:
    print("not all T5LayerNorm are different")

if Embedding_index == Embedding_diff_index:
    print("all Embeddings are different")
else:
    print("not all Embeddings are different")