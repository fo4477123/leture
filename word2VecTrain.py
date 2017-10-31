# -*- coding: utf-8 -*-

from gensim.models import word2vec,TfidfModel
from gensim.models.keyedvectors import KeyedVectors

import logging
import random
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import SQLAcess as sqlAccess

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=500,workers=8)
    doc = open("wiki_seg.txt",encoding = 'utf8')    
    # Save our model.
    model.save("med250.model3.bin")
    
    print("test")
    

if __name__ == "__main__":
    main()
