import csv
import os

import pandas as pd
import numpy as np
from collections import Counter
from itertools import product


def read_fasta(file_path):
    with open(file_path, 'r') as file:
        sequences = {}
        sequence_name = None
        sequence_data = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):  # 这是一个新的序列的开始
                if sequence_name:  # 保存前一个序列
                    sequences[sequence_name] = ''.join(sequence_data)
                sequence_name = line[1:]  # 去除'>'字符
                sequence_data = []
            else:
                sequence_data.append(line)
        # 保存最后一个序列
        if sequence_name:
            sequences[sequence_name] = ''.join(sequence_data)
    return sequences


# 第一个特征
def calculate_gc_content(sequence):
    gc_count = sum(nucleotide in 'GC' for nucleotide in sequence)  # G和C的总数
    sequence_length = len(sequence)  # 序列长度
    return gc_count / sequence_length


def save_gc_content_to_csv(fasta_sequences, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for name, sequence in fasta_sequences.items():
            gc_content = calculate_gc_content(sequence)
            writer.writerow([gc_content])  # 写入GC含量


# 第二个特征
def save_sequence_lengths_to_csv(fasta_sequences, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for name, sequence in fasta_sequences.items():
            sequence_lengths = len(sequence)
            writer.writerow([sequence_lengths])  # 写入序列长度


# 第三个特征
def generate_all_kmers(k, bases='ACGT'):
    # 生成所有可能的K-mer组合
    return {''.join(kmer) for kmer in product(bases, repeat=k)}


def calculate_kmer_frequencies(sequence, k, all_kmers):
    # 计算序列中每个K-mer的出现次数，如果K-mer不存在则为0
    kmer_counts = Counter((sequence[i:i + k] for i in range(len(sequence))))
    kmer_frequencies = {kmer: (kmer_counts[kmer] if kmer in kmer_counts else 0) for kmer in all_kmers}
    return kmer_frequencies


def save_kmer_frequencies_to_csv(sequences, k_values, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # 为每个K值生成所有可能的K-mer
        all_kmers_set = {k: generate_all_kmers(k) for k in k_values}

        # 写入每个序列的K-mer频率
        for sequence_name, sequence in sequences.items():
            row = []
            for k in k_values:
                all_kmers = all_kmers_set[k]
                kmer_frequencies = calculate_kmer_frequencies(sequence, k, all_kmers)
                # 添加每个K-mer的频率到行
                row.extend([str(kmer_frequencies[kmer]) for kmer in sorted(all_kmers)])
            writer.writerow(row)


def kcsv(csv_file_path, output_path):
    # 读取CSV文件
    with open(csv_file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        # 存储计算结果的列表
        row_ratios = []
        for row in csv_reader:
            # 将行中的每个元素转换为浮点数
            row_floats = [float(x) for x in row]

            # 计算每个元素占所在行的比重
            row_sum = sum(row_floats)
            row_ratios.append([x / row_sum for x in row_floats])

        # 如果需要将结果写回CSV文件
    with open(output_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        for row in row_ratios:
            csv_writer.writerow(row)


def w2v(w2v_file_path):
    # cdhit1
    # file_line_counts = [3659, 2188, 2685, 2566, 3871, 4239, 1963, 2511, 2666, 4602]
    # cdhit0
    file_line_counts = [4046, 2550, 3822, 3069, 4717, 4240, 2420, 4000, 3000, 4820]
    name = ['ATH1', 'CSA1', 'SLY1', 'STU1', 'ZMA1', 'ATH0', 'CSA0', 'SLY0', 'STU0', 'ZMA0']

    # 计算每个文件的起始行和结束行
    with open(w2v_file_path, 'r') as file:
        reader = csv.reader(file)
        current_line = 0
        file_index = 0

        # 遍历每个期望行数
        for i in range(10):
            # 打开新的 CSV 文件
            output_file_path = rf'data/features_cdhit0/{name[i]}_w2v.csv'
            with open(output_file_path, 'w', newline='') as output_file:
                writer = csv.writer(output_file)

                # 写入数据直到达到期望的行数
                for _ in range(file_line_counts[i]):
                    row = next(reader)
                    writer.writerow(row)
                    current_line += 1

            # 更新文件索引
            file_index += 1


if __name__ == '__main__':
    plants = ['ATH', 'CSA', 'SLY', 'STU', 'ZMA']
    k_values = [1, 2, 3, 4]  # K-mer的值
    # 计算去冗余前的数据量
    # for plant in plants:
    #     x = read_fasta(rf'data/NONCODEv6/NONCODEv6_{plant}.fa')
    #     print(len(x.values()))
    # 计算去冗余后的数据量
    # for plant in plants:
    #     x = read_fasta(rf'data/cd_hit_/NCBI_{plant}.fasta')
    #     print(len(x.values()))

    # fasta_sequences = {}
    # for plant in plants:
    #     fasta_sequences.update(read_fasta(rf'data/cd_hit_/NCBI_{plant}.fasta'))
    # save_gc_content_to_csv(fasta_sequences, 'm_gc.csv')
    # save_sequence_lengths_to_csv(fasta_sequences, 'm_len.csv')
    #
    # save_kmer_frequencies_to_csv(fasta_sequences, k_values, 'm_kmer_frequencies.csv')
    # kcsv('m_kmer_frequencies.csv', 'm_kmer.csv')
    # for plant in plants:
    #     x = read_fasta(rf'data/NONCODEv6/NONCODEv6_{plant}.fa')
    #     print(x.values())
    w2v('cdhit0w2v.csv')
    # 合并
    # x = read_fasta(rf'data/cd_hit_/i.fa')
    # print(len(x.values()))
    #
    # fasta_sequences = read_fasta(rf'data/cd_hit_/i.fa')
    # save_gc_content_to_csv(fasta_sequences, 'Inc1_gc.csv')
    # save_sequence_lengths_to_csv(fasta_sequences, 'Inc1_len.csv')
    #

    # save_kmer_frequencies_to_csv(fasta_sequences, k_values, 'Inc1_kmer_frequencies.csv')
    # kcsv('Inc1_kmer_frequencies.csv', 'Inc1_kmer.csv')

    # 单独计算 拟南芥lncRNA 和 玉米mRNA
    # fasta_sequences = read_fasta(rf'data/cd_hit_/NONCODEv6_ATH.fa')
    # save_gc_content_to_csv(fasta_sequences, 'ATH_gc.csv')
    # save_sequence_lengths_to_csv(fasta_sequences, 'ATH_len.csv')
    # save_kmer_frequencies_to_csv(fasta_sequences, k_values, 'ATH_kmer_frequencies.csv')
    # kcsv('ATH_kmer_frequencies.csv', 'ATH_kmer.csv')

    # fasta_sequences = read_fasta(rf'data/cd_hit_/ml.fa')
    # save_gc_content_to_csv(fasta_sequences, 'ZMA_gc.csv')
    # save_sequence_lengths_to_csv(fasta_sequences, 'ZMA_len.csv')
    # save_kmer_frequencies_to_csv(fasta_sequences, k_values, 'ZMA_kmer_frequencies.csv')
    # kcsv('ZMA_kmer_frequencies.csv', 'ZMA_kmer.csv')

    # print(len(fasta_sequences.values()))
