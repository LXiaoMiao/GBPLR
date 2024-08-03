from extract import *
from select1 import *
from classifer import *
import pandas as pd


# 随机从 15981*343 的 m1.csv，取与正例同num条数据
def suiji(num):
    m1 = pd.read_csv('m1.csv', header=None)
    random_sample = m1.sample(n=num)
    print(random_sample)
    random_sample.to_csv(f'm1_random_{num}.csv', index=False, header=False)


# 计算每个植物的lncRNA和mRNA共的三个特征，得到5*2*3个特征文件
def all_extract(plant, label):
    if label == 1:
        fasta = read_fasta(rf'data/NONCODEv6/NONCODEv6_{plant}.fa')
    else:
        fasta = read_fasta(rf'data/NCBI/NCBI_{plant}.fasta')
    save_gc_content_to_csv(fasta, rf'data/features_cdhit0/{plant}{label}_gc.csv')
    save_sequence_lengths_to_csv(fasta, rf'data/features_cdhit0/{plant}{label}_len.csv')
    k_values = [1, 2, 3, 4]  # K-mer的值
    save_kmer_frequencies_to_csv(fasta, k_values, rf'data/features_cdhit0/{plant}{label}_kmer_frequencies.csv')
    kcsv(rf'data/features_cdhit0/{plant}{label}_kmer_frequencies.csv', rf'data/features_cdhit0/{plant}{label}_kmer.csv')


# 卡方选择50个还是别的数量实验
def kafang(k):
    # 读取CSV文件
    df = pd.read_csv(rf'data/features_cdhit1/combined_ATH1_ZMA0.csv', header=None)
    # 将数据集分为特征和标签
    X = df.iloc[:, :-1]  # 特征
    y = df.iloc[:, -1]  # 标签

    # 使用卡方检验选择前k个特征
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)

    # 获取选中的特征的列索引
    selected_features = selector.get_support(indices=True)

    # 打印选中的特征的列索引
    print("Selected feature indices:", selected_features)

    # 将选中的特征和标签组合成一个新的DataFrame
    df_selected = pd.DataFrame(X_new, columns=[f'Feature_{i + 1}' for i in selected_features])
    df_selected['Label'] = y

    # 保存新的DataFrame到CSV文件
    df_selected.to_csv(rf'data/kafang/selected_{k}.csv', header=None, index=False)
    return selected_features


if __name__ == '__main__':
    plants = ['ATH', 'CSA', 'SLY', 'STU', 'ZMA']
    # cdhit0
    lncs = {'ATH': 4046, 'CSA': 2550, 'SLY': 3822, 'STU': 3069, 'ZMA': 4717}
    ms = {'ATH': 4240, 'CSA': 2420, 'SLY': 4000, 'STU': 3000, 'ZMA': 4820}
    # cdhit1
    # lncs = {'ATH': 3659, 'CSA': 2188, 'SLY': 2685, 'STU': 2566, 'ZMA': 3871}
    # ms = {'ATH': 4239, 'CSA': 1963, 'SLY': 2511, 'STU': 2666, 'ZMA': 4602}

    # lnc, m = 'ATH_glk', 'm1_random_3659'
    # suiji(3659)
    # combine_ml(lnc, m)
    # select_feature(lnc, m)
    # train_model(lnc, m)

    # 计算三个特征
    # for plant in plants:
    #     for label in [0, 1]:
    #         all_extract(plant, label)

    # 将每四个特征文件融合为一个文件
    # for lnc in lncs.keys():
    #     combine_features(lnc, lncs[lnc], 1)
    # for m in ms.keys():
    #     combine_features(m, ms[m], 0)

    # 4.1
    # header = [' ', 'acc', 'pre', 'rec', 'f1']
    # print(' ')
    # res = train_model('selected_ATH_glk_m1_random_3659.csv')
    # df = pd.DataFrame(res)
    # df.to_csv(f'output.csv', index=False, header=header)

    # 4.3
    # for lnc in [i+'1' for i in plants]:
    #     for m in [i + '0' for i in plants]:
    #         header = [f'{lnc}_{m}', 'acc', 'pre', 'rec', 'f1']
    #         print(f'{lnc}_{m}')
    #         combine_ml(lnc, m)
    #         fe = select_feature(lnc, m)
    #         res = train_model(lnc, m)
    #         df = pd.DataFrame(res)
    #         df.to_csv(rf'data/features_cdhit0/output_{lnc}_{m}.csv', index=False, header=header)
    # print(fe, res)
    
    # 卡方
    ks = [20, 30, 40, 50, 100]
    for k in ks:
        header = [f'{k}', 'acc', 'pre', 'rec', 'f1']
        kafang(k)
        res = train_model(k)
        df = pd.DataFrame(res)
        df.to_csv(rf'data/kafang/output_{k}.csv', index=False, header=header)
