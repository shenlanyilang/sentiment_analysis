# -*- coding:utf-8 -*-
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
from opencc import OpenCC
import re
import jieba
from keras.utils import to_categorical
from typing import List,Dict
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from utils import fan2jian


OP = OpenCC('t2s')
zh_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')


class WordDictionary(object):
    def __init__(self, w2v):
        self.word2idx = dict()
        self.idx2word = list()
        self.word2idx['<pad>'] = 0
        self.idx2word.append('<pad>')
        for i, word in enumerate(w2v.index2word, start=1):
            self.word2idx[word] = i
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


class Processor(object):
    def __init__(self, config):
        self.dictionary = None
        self.w2v_model = None
        self.config = config

    def init(self, w2v_path):
        w2v_model = Word2VecKeyedVectors.load_word2vec_format(w2v_path,
                                                              binary=False,
                                                            unicode_errors='ignore',
                                                              encoding='utf8')
        self.w2v_model = w2v_model
        self.dictionary = WordDictionary(self.w2v_model)

    def to_embedding(self):
        idx2word = self.dictionary.idx2word
        size = len(idx2word)
        dim = len(self.w2v_model[idx2word[-1]])
        embedding_matrix = np.zeros(shape=(size, dim))
        for i, word in enumerate(idx2word[1:], start=1):
            embedding_matrix[i] = self.w2v_model[word]
        return embedding_matrix

    def get_features(self, df: pd.DataFrame):
        contents = df[self.config['feature_column']].apply(lambda x: fan2jian(x))\
            .tolist()
        contents = [self._cut_sent(content) for content in contents]
        contents = [[word for word in words if zh_pattern.search(word)]
                    for words in contents]
        sequences = [self._to_sequences(content) for content in contents]
        features = padding_examples(sequences, self.config['seq_len'])
        return features

    @staticmethod
    def get_labels(df: pd.DataFrame, column, label2idx):
        labels = df[column].tolist()
        label_ids = [label2idx[la] for la in labels]
        return to_categorical(label_ids)

    def _to_sequences(self,content:List[str]):
        word2idx = self.dictionary.word2idx
        return [word2idx[word] for word in content if word in word2idx]

    @staticmethod
    def _cut_sent(sent):
        return list(jieba.cut(sent))


class Label(object):
    def __init__(self, labels, name):
        labels_uniq = set(labels)
        self.name = name
        self.label2idx = {}
        self.idx2label = []
        for i, la in enumerate(labels_uniq):
            self.label2idx[la] = i
            self.idx2label.append(la)


def padding_examples(sequences: List[List[int]], max_len=50):
    return pad_sequences(sequences, maxlen=max_len)


def grade_map(labels:List):
    grade2idx = dict()
    idx2grade = dict()
    grade_uniq = list(set(labels))
    grade_uniq.sort()
    for i,grade in enumerate(grade_uniq):
        grade2idx[grade] = i
        idx2grade[i] = grade
    return grade2idx, idx2grade
