from gensim.corpora import Dictionary
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from gensim.models.fasttext import FastText


def tf_idf(sent):   
    word2vec_model = FastText.load_fasttext_format(r'D:\20192\machine_learning\wiki.vi.bin',encoding='utf-8')
    dct = Dictionary(sent)
    corpus = [dct.doc2bow(line) for line in sent]
    tf_idf_model = TfidfModel(corpus)
    vector = tf_idf_model[corpus]
    d = {dct.get(id): value for doc in vector for id, value in doc}
    sents_emd = []
    no_of_sent = sum(1 for i in sent)
    for i in xrange(no_of_sent):
        sent_emd = []
        for j in xrange(len(sent[i])):
            word = sent[i][j]
            if word in word2vec_model:
                emd = d[word]*word2vec_model[word]
                sent_emd.append(emd)
        sent_emd_np = np.array(sent_emd)
        sum_ = sent_emd_np.sum(axis=0)
        result = sum_/np.sqrt((sum_**2).sum())
        sents_emd.append(result)  
    return sents_emd
