import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier


def read_file(k):
    # load the csv file as a dataframe
    # df = pd.read_csv(rf'data/features_cdhit0/selected_{lnc}_{m}.csv', header=None)
    df = pd.read_csv(rf'data/kafang/selected_{k}.csv', header=None)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    # ensure input data is floats
    X = X.astype('float32')
    # label encode target and ensure the values are floats
    y = LabelEncoder().fit_transform(y)
    y = y.astype('float32')
    y = y.reshape((len(y), 1))
    return X, y


def train_model(k):
    res = []
    X, y = read_file(k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Bayes
    clf = GaussianNB()
    clf = clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print('Bayes')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)
    res.append(['Bayes', acc, precision, recall, f1])

    # SVM
    # clf = SVC(kernel='linear')
    # clf = clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    # y_pred = clf.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # print('SVM')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)

    # DecisionTree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print('DecisionTree')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)
    res.append(['DecisionTree', acc, precision, recall, f1])

    # RandomForest
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=12, random_state=0)
    clf = clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print('RandomForest')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)
    res.append(['RandomForest', acc, precision, recall, f1])

    # Bagging
    clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    clf = clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print('Bagging')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)
    res.append(['Bagging', acc, precision, recall, f1])

    # AdaBoost
    clf = AdaBoostClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print('AdaBoost')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)
    res.append(['AdaBoost', acc, precision, recall, f1])

    # GBDT
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)
    clf = clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print('GBDT')
    # print('acc: ', acc)
    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)
    res.append(['GBDT', acc, precision, recall, f1])

    return res


# data
if __name__ == '__main__':
    train_model()
