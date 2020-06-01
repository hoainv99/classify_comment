
import pandas as pd
import re
import smart_open
import string
import sys
import csv
from nltk import word_tokenize
# from underthesea import word_tokenize
#get data has one column
def get_data_train(path):
    data=pd.read_csv(path,header=None, error_bad_lines=False)
    #get vietnamese stopword then save in list stopwords

    stopwords=[]
    data_stopword=pd.read_csv('vietnamese-stopwords.txt',header=None)
    for l in data_stopword[0]:
        stopwords.append(l)
    cnt=0
    remove = ['(', ')', '^','"','?','!','.','❤️',':','T^T']
    all_docs=[]
    dem=0
    #get data into all_docs
    for line in data[0]:
        if cnt%3!=1:
            cnt+=1
            continue
        cnt+=1
        words=word_tokenize(line)
        #words=[w for w in words if w not in stopwords]  
        words=[w for w in words if w not in remove]  
        words=[w.lower() for w in words]
        #words=[no_accent_vietnamese(w) for w in words]
        words=[w for w in words if w.isalpha()]
        all_docs.append(words)
    all_labels=[]
    cnt=0
    for line in data[0]:
        if cnt%3!=2:
            cnt+=1
            continue
        cnt+=1
        all_labels.append(int(line))
    return all_docs,all_labels
def get_data_test(path):
    data=pd.read_csv(path,header=None, error_bad_lines=False)
    #get vietnamese stopword then save in list stopwords
    cnt=0
    remove = ['(', ')', '^','"','?','!','.','❤️',':','T^T']
    all_docs=[]
    dem=-2
    #get data into all_docs
    for line in data[0]:
        if cnt%2!=1:
            cnt+=1
            continue
        cnt+=1
        dem+=1
        words=word_tokenize(line)
        #words=[w for w in words if w not in stopwords]  
        words=[w for w in words if w not in remove]  
        words=[w.lower() for w in words]
        words=[w for w in words if w.isalpha()]
        if words!=[]:
            all_docs.append(words)
    return all_docs





