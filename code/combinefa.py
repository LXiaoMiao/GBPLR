import os
from extract import read_fasta


# 合并在ncbi下载的mRNA数据集
def merge_fasta_files(folder_path, output_file):
    # 初始化一个列表，用于存储所有fasta文件的内容
    fasta_contents = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.fasta'):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, filename)

            # 打开并读取文件内容
            with open(file_path, 'r') as file:
                fasta_contents.append(file.read())

    # 将所有内容写入新文件
    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(fasta_contents))


# 使用示例
plants = ['ATH', 'CSA', 'SLY', 'STU', 'ZMA']
# for plant in plants:
#     folder_path = rf'data\NCBI\{plant}'  # 替换为你的文件夹路径
#     output_file = rf'data\NCBI\NCBI_{plant}.fasta'  # 合并后的文件名
#     merge_fasta_files(folder_path, output_file)

# 测试合并后的数据量与下载的是否相符
for plant in plants:
    x = read_fasta(rf'data\NCBI\NCBI_{plant}.fasta')
    print(len(x.values()))
