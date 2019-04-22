import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 以空格来划分每一个分词
def preprocess(path):
    text_with_space = ""
    textfile = open(path, "r", encoding="utf-8").read()
    textcute = jieba.cut(textfile)
    for word in textcute:
        text_with_space += word + " "
        # print(text_with_space)
    return text_with_space



def loadtrainset(path, classtag):
    allfiles = os.listdir(path)  # os.path.isdir()用于判断对象是否为一个目录，并返回此目录下的所有文件名
    processed_textset = []
    allclasstags = []

    for thisfile in allfiles:
        # print(thisfile)
        path_name = path + "/" + thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    return processed_textset, allclasstags  # 数组形式--processed_textset 文件的具体内容， allclasstags 文件分类


processed_textdata1, class1 = loadtrainset("E:/Desk/MyProjects/Python/NLP/dataset/train/hotel", "宾馆")
processed_textdata2, class2 = loadtrainset("E:/Desk/MyProjects/Python/NLP/dataset/train/travel", "旅游")


train_data = processed_textdata1 + processed_textdata2
# print(train_data)  # 前半部分是宾馆， 后半部分是旅游 train
classtags_list = class1 + class2  # 前半部分是宾馆， 后半部分是旅游 train 集结果
#
# print(train_data)
# print(classtags_list)
"""
# CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵
    get_feature_names()可看到所有文本的关键字
    vocabulary_可看到所有文本的关键字和其位置
    toarray()可看到词频矩阵的结果

"""

count_vector = CountVectorizer()
vecot_matrix = count_vector.fit_transform(train_data)

# print (count_vector.get_feature_names ())  #看到所有文本的关键字
# print (count_vector.vocabulary_)   #文本的关键字和其位置
# print (vecot_matrix.toarray ())  #词频矩阵的结果

# #TFIDF
"""TfidfTransformer是统计CountVectorizer中每个词语的tf-idf权值
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
vectorizer.fit_transform(corpus)将文本corpus输入，得到词频矩阵

将这个矩阵作为输入，用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵
TfidfTransformer + CountVectorizer  =  TfidfVectorizer

这个成员的意义是词典索引，对应的是TF-IDF权重矩阵的列，只不过一个是私有成员，一个是外部输入，原则上应该保持一致。
    use_idf：boolean， optional 启动inverse-document-frequency重新计算权重
"""
# print(train_tfidf)  # vecot_matrix输入，得到词频矩阵
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vecot_matrix)
#MultinomialNB(),fit() ,多分类， Fit Naive Bayes classifier according to X,根据 X，Y，结果 类别，进行多分类
# print(train_tfidf)
# print(classtags_list)
clf = MultinomialNB().fit(train_tfidf, classtags_list)

testset = []
#  path = "E:\Desk\MyProjects\Python/NB_text\dataset\train" 测试文本
path = "E:/Desk/MyProjects/Python/NLP/dataset/tt"  # 测试此路径下的各文件属于哪个类别
allfiles = os.listdir(path)
hotel = 0
travel = 0


for thisfile in allfiles:
    path_name = path + "/" + thisfile  # 得到此目录下的文件绝对路径
    new_count_vector = count_vector.transform([preprocess(path_name)]) # 得到测试集的词频矩阵
# 用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵
    new_tfidf = TfidfTransformer(use_idf=False).fit_transform(new_count_vector)

    # 根据 由训练集而得到的分类模型，clf ,由 测试集的 TF-IDF权重矩阵来进行预测分类
    predict_result = clf.predict(new_tfidf)
    print(predict_result)
    print(thisfile)

    if(predict_result == "宾馆"):
        hotel += 1
    if(predict_result == "旅游"):
        travel += 1

print("宾馆" + str(hotel))
print("旅游" + str(travel))
