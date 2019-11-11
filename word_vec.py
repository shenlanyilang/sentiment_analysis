# -*- coding:utf-8 -*-
import jieba
import pandas as pd
from typing import List, Dict
import codecs
import os
from gensim.models.word2vec import Word2Vec
from utils import fan2jian
import re
import json

zh_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')

def load_data(path):
    data = pd.read_csv(path)
    return data

def load_vocab(path):
    vocab = []
    with codecs.open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line.split('\t')) == 2:
                word,cnt = line.split('\t')
                vocab.append((word, str(cnt)))
    return vocab

def gen_char_vocab(sentences:List[str], vocab_size=5000):
    counts = {}
    vocab = []
    vocab_path = './data/vocab.txt'
    if os.path.exists(vocab_path):
        print('vocab already exists!')
        vocab = load_vocab(vocab_path)
        return vocab
    for sent in sentences:
        sent = sent.replace('\r\n','\n')
        sent = sent.replace('\n','')
        sent = sent.replace(' ','')
        for char in sent:
            if len(char.strip()) == 0:
                continue
            counts[char] = counts.get(char, 0) + 1
    i = 0
    for word, cnt in sorted(list(counts.items()), key=lambda x: x[1],
                            reverse=True):
        if i >= vocab_size:
            break
        i += 1
        vocab.append((word, cnt))
    save_vocab(vocab, vocab_path)
    return vocab

def gen_word_vocab(sentences:List[List[str]], vocab_size=50000):
    counts = {}
    vocab = []
    vocab_path = './data/word_vocab.txt'
    if os.path.exists(vocab_path):
        print('vocab already exists!')
        vocab = load_vocab(vocab_path)
        return vocab
    for sent in sentences:
        for word in sent:
            counts[word] = counts.get(word, 0) + 1
    i = 0
    for word, cnt in sorted(list(counts.items()), key=lambda x: x[1],
                            reverse=True):
        if i >= vocab_size:
            break
        i += 1
        vocab.append((word, cnt))
    save_vocab(vocab, vocab_path)
    return vocab

def save_vocab(vocab, path):
    with codecs.open(path, 'w') as f:
        for word,cnt in vocab:
            f.write(word + '\t' + str(cnt) + '\n')
    print('vocab write finished')


def filter_sent(sentence, vocab):
    res = []
    for char in sentence:
        if char in vocab:
            res.append(char)
        else:
            res.append('<UNK>')
    return res

def filter_words(sents:List[str]):
    filtered = [word for word in sents if zh_pattern.search(word)]
    return filtered

def get_word_embd(line):
    word_embd = line.split()
    word = word_embd[0]
    embd = word_embd[1:]
    return word, embd

def write_lines(data, path):
    with codecs.open(path, 'w') as f:
        f.write('\n'.join(data))

def write_json(data, path):
    with codecs.open(path, 'w') as f:
        json.dump(data, f)

def gen_small_word2vec(vocab, embedding_file_path):
    new_vocab = {}
    small_embedding = []
    with codecs.open(embedding_file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            word, embd = get_word_embd(line)
            if word in vocab:
                new_vocab[word] = vocab[word]
                small_embedding.append(line)
                del vocab[word]
            if len(vocab) == 0:
                break
    print('totally words in new vocab : {}'.format(len(new_vocab)))
    write_lines(small_embedding,'./data/sm_word_emb.wiki')
    write_json(new_vocab, './new_vocab.txt')


if __name__ == '__main__':
    # data = load_data('/home/student/dataset/project-3/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')
    # sents = data['content'].tolist()
    # sents = [fan2jian(content) for content in sents]
    # vocab = dict(gen_char_vocab(sents, vocab_size=5000))
    # raw_sents = [list(sent) for sent in sents]
    # filtered_sents = [filter_sent(sent, vocab) for sent in raw_sents]
    # w2v_model = Word2Vec(filtered_sents, size=100, window=5,min_count=1,
    #                      workers=4, iter=10)
    # w2v_model.wv.save_word2vec_format('./data/w2v.vector', binary=True)
    #
    # data = load_data('/home/student/dataset/project-3/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')
    # sents = data['content'].tolist()
    # sents = [fan2jian(content) for content in sents]
    # all_words = [filter_words(list(jieba.cut(sent))) for sent in sents]
    # vocab = dict(gen_word_vocab(all_words, vocab_size=50000))
    # filtered_sents = [filter_sent(sent, vocab) for sent in all_words]
    # w2v_model = Word2Vec(filtered_sents, size=100, window=5, min_count=1,
    #                      workers=4, iter=10)
    # w2v_model.wv.save_word2vec_format('./data/w2v_word.vector', binary=True)

    vocab = dict(gen_word_vocab([], vocab_size=50000))
    gen_small_word2vec(vocab, '/home/student/project/project-03/nlp_strong/sgns.wiki.word')
