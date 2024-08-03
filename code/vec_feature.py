import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# 使用gensim训练Char2Vec模型
def train_char2vec(sequences, vector_size=100, window=5, min_count=1, epochs=10):
    # 将每个序列转换为gensim的句子格式
    sentences = [list (seq) for seq in sequences]
    model = Word2Vec (sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    return model


# 为每个序列提取特征向量
def extract_features(model, sequences):
    feature_vectors = []
    for sequence in sequences:
        # 将序列转换为字符列表
        sequence_chars = list (sequence)
        # 计算每个字符的向量表示的平均值
        feature_vector = np.mean ([model.wv[char] for char in sequence_chars if char in model.wv], axis=0)
        feature_vectors.append (feature_vector)
    return np.array (feature_vectors)


def read_fasta(file_path):
    with open (file_path, 'r') as file:
        sequences = {}
        sequence_name = None
        sequence_data = []
        for line in file:
            line = line.strip ()
            if line.startswith ('>'):  # 这是一个新的序列的开始
                if sequence_name:  # 保存前一个序列
                    sequences[sequence_name] = ''.join (sequence_data)
                sequence_name = line[1:]  # 去除'>'字符
                sequence_data = []
            else:
                sequence_data.append (line)
        # 保存最后一个序列
        if sequence_name:
            sequences[sequence_name] = ''.join (sequence_data)
    return sequences


if __name__ == '__main__':
    x = read_fasta (rf'nocdhit.fa')
    # print(list(x.values()))
    rna_sequences = list (x.values ())
    # 训练Char2Vec模型
    model = train_char2vec (rna_sequences, vector_size=100, window=5, min_count=1, epochs=10)
    # 提取RNA序列的特征向量
    features = extract_features (model, rna_sequences)
    features = pd.DataFrame(features)
    scaler = MinMaxScaler ()
    features.iloc[:, :]=scaler.fit_transform (features.iloc[:, :])
    # 打印特征向量数组的形状
    print (features.shape)

    pd.DataFrame (features).to_csv ("nocdrna_features.csv", index=False, header=None)
