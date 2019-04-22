import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯
from sklearn.externals import joblib
import time


def preprocess(path):
    text_with_space = "" # 以空格分开
    textfile = open(path, "r", encoding="utf-8").read()
    textcute = jieba.cut(textfile)  # 用 jieba 分词
    for word in textcute:
        text_with_space += word + " "
    return text_with_space

# word = preprocess('E:/Desk/MyProjects/Python/NB_text/text_demo')
# print(word)
path = 'E:/Desk/MyProjects/Python/NB_text'
allfiles = os.listdir (path)
# print(allfiles)


def loadtrainset(path, classtag):
    # 得到此目录下的所有文件夹
    allfiles = os.listdir (path)  # os.path.isdir()用于判断对象是否为一个目录，并返回此目录下的所有文件名
    processed_textset = []
    allclasstags = []

    for thisfile in allfiles:
        print (thisfile)
        path_name = path + "/" + thisfile
        processed_textset.append (preprocess (path_name))
        allclasstags.append(classtag)
    return processed_textset, allclasstags

path = 'E:/Desk/MyProjects/Python/NB_text/dataset/train/hotel'
classtag = 'hotel'
p,c = loadtrainset(path, classtag)
print(p)
# print(c)