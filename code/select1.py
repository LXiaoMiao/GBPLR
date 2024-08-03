import csv

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

from extract import read_fasta


# 分别将IncRNA和mRNA的数据集合并，得出两个csv
def combine_data(input_):
    plants = ['ATH', 'CSA', 'SLY', 'STU', 'ZMA']
    fasta_sequences = {}
    fasta_sequences.update(read_fasta(f'data/{input_}/m.fasta'))
    print(len(fasta_sequences))
    # if input_ == 'NCBI':
    #     files = [rf'data/NCBI/NCBI_{plant}.fasta' for plant in plants]
    #     output_file = 'data/NCBI/mRNA.fasta'
    # else:
    #     files = [rf'data/NONCODEv6/NONCODEv6_{plant}.fa' for plant in plants]
    #     output_file = 'data/NONCODEv6/IncRNA.fa'
    #
    # with open(output_file, 'w') as outfile:
    #     for file in files:
    #         with open(file, 'r') as infile:
    #             # 首先复制文件头，直到遇到空行
    #             for line in infile:
    #                 if line.strip() == '':
    #                     break
    #                 outfile.write(line)
    #             # 然后复制序列，直到再次遇到文件头
    #             while True:
    #                 line = infile.readline()
    #                 if line.startswith('>'):
    #                     break
    #                 outfile.write(line)
    #             # 跳过文件头，继续读取序列
    #             while True:
    #                 line = infile.readline()
    #                 if not line or line.startswith('>'):
    #                     break
    #                 outfile.write(line)
    #             # 如果不是最后一个文件，写入一个分隔符
    #             if file != files[-1]:
    #                 outfile.write("\n>merged_sequence\n")


# 将四个特征csv合并为一个csv并添加label
def combine_features(plant, row, label):
    # 读取CSV文件
    gc = pd.read_csv(rf'data/features_cdhit0/{plant}{label}_gc.csv', header=None)
    len_ = pd.read_csv(rf'data/features_cdhit0/{plant}{label}_len.csv', header=None)
    kmer = pd.read_csv(rf'data/features_cdhit0/{plant}{label}_kmer.csv', header=None)
    w2v = pd.read_csv(rf'data/features_cdhit0/{plant}{label}_w2v.csv', header=None)

    # mRNA=15981行 IncRNA=14969  m1=15897  Inc1=14963 拟南芥lncRNA=3659  玉米mRNA=4602
    # 创建一个空的DataFrame，具443列，初始值为NaN
    combined_df = pd.DataFrame(index=range(row), columns=range(443))

    # gc和len的数据填充到相应的列中
    combined_df.iloc[:, 0] = gc.iloc[:, 0]
    combined_df.iloc[:, 1] = len_.iloc[:, 0]

    # 将kmer的数据填充到剩余的列中
    for i in range(340):
        combined_df.iloc[:, i + 2] = kmer.iloc[:, i]

    # 将w2v的数据填充到剩余的列中
    for i in range(100):
        combined_df.iloc[:, i + 342] = w2v.iloc[:, i]

    # mRNA=0 IncRNA=1
    combined_df.iloc[:, -1] = label

    # 保存合并后的DataFrame到CSV文件，不包含表头
    combined_df.to_csv(rf'data/features_cdhit0/{plant}{label}.csv', index=False, header=False)


# 将两份包含四个特征和label的csv合并，后打乱
def combine_ml(lnc, m):
    # 读取CSV文件
    lnc_df = pd.read_csv(rf'data/features_cdhit0/{lnc}.csv', header=None)
    m_df = pd.read_csv(rf'data/features_cdhit0/{m}.csv', header=None)
    # w2v_df = pd.read_csv(f'w2v.csv', header=None)

    # 合并两个DataFrame
    combined_df = pd.concat([lnc_df, m_df], ignore_index=True)
    # combined_df = pd.concat([w2v_df, combined_df], axis=1, ignore_index=True)

    # 打乱DataFrame的行顺序
    np.random.seed(0)  # 为了可复现性，设置随机种子
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # 保存合并并打乱后的DataFrame到CSV文件
    combined_df.to_csv(rf'data/features_cdhit0/combined_{lnc}_{m}.csv', header=False, index=False)


# 卡方检验
def select_feature(lnc, m):
    # 读取CSV文件
    df = pd.read_csv(rf'data/features_cdhit0/combined_{lnc}_{m}.csv', header=None)

    # 将数据集分为特征和标签
    X = df.iloc[:, :-1]  # 特征
    y = df.iloc[:, -1]  # 标签

    # 使用卡方检验选择前50个特征
    selector = SelectKBest(chi2, k=50)
    X_new = selector.fit_transform(X, y)

    # 获取选中的特征的列索引
    selected_features = selector.get_support(indices=True)

    # 打印选中的特征的列索引
    print("Selected feature indices:", selected_features)

    # 将选中的特征和标签组合成一个新的DataFrame
    df_selected = pd.DataFrame(X_new, columns=[f'Feature_{i + 1}' for i in selected_features])
    df_selected['Label'] = y

    # 保存新的DataFrame到CSV文件
    df_selected.to_csv(rf'data/features_cdhit0/selected_{lnc}_{m}.csv', header=None, index=False)
    return selected_features


if __name__ == '__main__':
    ml = 'm'
    ly = 1
    glk = 'glk'
    # combine_features(ml)
    # combine_ml(ly, glk)
    # select_feature(ly, glk)
    # combine_data('NCBI')
    # combine_data('NONCODEv6')

    # 合并 拟南芥lncRNA 和 玉米mRNA
    # combine_features('ZMA', '', glk)
    # combine_ml('', glk)
    # select_feature('ATHZMA', 'glk')
    combine_features('ATH', 4240, 0)
