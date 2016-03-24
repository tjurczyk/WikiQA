from collections import defaultdict
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
import sys
import copy
from os.path import exists
from os import makedirs
from pprint import pprint
from representation.wordrepresentation import Word2VecModel as w2vec
from representation.question import Question
from decomposition.sentence import get_sentence_words, get_normalized_numbers
from collections import defaultdict
import logging
import re
import itertools
import globals
q_limit = None
# def get_sentence_words(sentence):
#     return sentence.strip().lower().split()

def default_word():
    return 'UNK'
def default_vocab():
    return 2

def parse():
    print("loading embeddings")
    # embedding = Word2Vec.load_word2vec_format('/home/mike/data/GoogleNews-vectors-negative300.bin', binary=True)
    # dim = 300
    # embedding = Word2Vec.load_word2vec_format('/home/mike/data/wiki_nyt.skip.simplified.low.200.vectors.bin', binary=True)
    # dim = 200
    corpus = '/home/mike/Downloads/wiki_nyt.simplified.low.corpus'
    # word2vocab = defaultdict(default_vocab)     
    # vocab2word = defaultdict(default_word)
    # word_set = set(line.strip() for line in open(corpus))
    # print('vocab size ',len(word_set))
    # vocab_size = 3
    # known = 0
    # unknown = 0

    # vocab2word[0] = ''
    # vocab2word[1] = '<END>'
    # vocab2word[2] = '<UNK>'

    # word2vocab[''] = 0
    # word2vocab['<END>'] = 1
    # word2vocab['<UNK>'] = 2

    # newEmbedding = [[0]*dim,[0]*dim,[0]*dim] # empty, end, unknown
    # for word in word_set:
    #     word = re.sub("\d", "0", word)
    #     '''
    #     if word in embedding:
    #         newEmbedding.append(embedding[word])
    #     else:
    #         newEmbedding.append([0]*dim)
    #         unknown += 1
    #     word2vocab[word] = vocab_size
    #     vocab2word[vocab_size] = word
    #     '''
    #     if word in embedding:
    #         newEmbedding.append(embedding[word])
    #         word2vocab[word] = vocab_size
    #         vocab2word[vocab_size] = word
    #         vocab_size +=1
    #     else:
    #         word2vocab[word] = 2
    #         unknown += 1

    # print vocab_size,unknown
    # with open('./triggering/data/embedding.pickle', 'wb') as f:
        # pickle.dump([np.asarray(newEmbedding,np.float32),word2vocab,vocab2word,vocab_size],f,protocol=2)
    with open('./triggering/data/embedding.pickle', 'r') as f:
        _,word2vocab,vocab2word,vocab_size = pickle.load(f)
    max_n_candidates = 30
    with open(corpus,'r') as f:
        for idx,_ in enumerate(f):
            pass
        n_questions = idx+1
    print(n_questions)
    maxlen = 64
    wikinyt = np.memmap('./triggering/data/wikinyt.nmap', dtype=np.uint16, mode="w+", shape=(n_questions,max_n_candidates,maxlen))
    wikinyt_mask = np.memmap('./triggering/data/wikinyt_mask.nmap', dtype=np.bool, mode="w+", shape=(n_questions,max_n_candidates,maxlen))

    # wikinyt = np.zeros((n_questions,max_n_candidates,maxlen),dtype=np.int32)
    # wikinyt_mask = np.zeros((n_questions,max_n_candidates,maxlen),dtype=np.int8)
    q_idx = 0
    with open(corpus,'r') as f:
        for idx,line in enumerate(f):
            add(line, word2vocab, wikinyt, wikinyt_mask, q_idx, idx%max_n_candidates)
            if (idx+1)%max_n_candidates == 0:
                q_idx+=1
        # with open('./triggering/data/wikinyt.npz', 'wb') as f:
            # np.savez(f,wikinyt=wikinyt, wikinyt_mask=wikinyt_mask)

def add(sentence, word2vocab, wikinyt, wikinyt_mask, q_idx, c_idx):
    for i,word in enumerate(sentence):
        if i == 64:
            break
        wikinyt[q_idx,c_idx,i] = word2vocab[re.sub("\d", "0", word.lower())]
        wikinyt_mask[q_idx,c_idx,i] = True
if __name__ == '__main__':
    parse()