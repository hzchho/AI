import numpy as np
import time
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 读取数据
def load_data(file_path):
    N=0
    with open(file_path, 'r', encoding='utf-8') as file:
        data=[]
        labels=[]
        a=0
        for line in file:
            N+=1
            if a==0:
                a=1
                continue
            parts = line.strip().split()
            data.append(' '.join(parts[3:]))  # 句子部分
            labels.append(int(parts[1]))  # 情感标签部分
        return data, labels, N
# 文本情感分类模型
def text_sentiment_classification(train_data, train_labels, test_data,N):
    # 特征提取
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)

    # 训练k-NN分类器
    knn = KNeighborsClassifier(n_neighbors=int(math.sqrt(2*N)))  # 设置k值为5
    knn.fit(X_train, train_labels)

    # 预测
    predictions = knn.predict(X_test)

    return predictions

# 加载训练数据和测试数据
time1=time.time()
train_data, train_labels , N= load_data('code/train.txt')  # 训练数据文件
test_data, test_labels , _= load_data('code/test.txt')  # 测试数据文件

# 情感分类
predictions = text_sentiment_classification(train_data, train_labels, test_data,N)

# 计算准确率
accuracy = accuracy_score(test_labels, predictions)
time2=time.time()
print("(该代码调用相关的库实现)")
print("Using function provide:")
print("The accuracy is:", accuracy)
print("Used time:",time2-time1,"s")
