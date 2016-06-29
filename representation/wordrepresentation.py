from gensim.models import Word2Vec
import numpy as np
from numpy.random import choice
from scipy.spatial.distance import cosine
import sys
import operator

import globals
import cPickle as pickle


class Word2VecModel:
    """
    Class for Word2Vec representation of a text
    """
    model = None

    def __init__(self, vocabulary, filename="../GoogleNews-vectors-negative300.bin"):
        if filename is not None:
            self.word_vecs = self.load_bin_vec(filename, vocabulary)
            self.d = len(self.word_vecs.values()[0])
            self.add_unknown_words(self.word_vecs, vocabulary, d=self.d)

    def load_bin_vec(self, filename, vocabulary):
        word_vectors = {}

        with open(filename, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            print("vocab size: %d, layer1_size: %d" % (vocab_size, layer1_size))

            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = ''.join(word)
                        break
                    if ch != b'\n':
                        word.append(ch)

                if word in vocabulary:
                    word_vectors[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)

        print("num words already in word2vec: %d" % len(word_vectors))
        print("vocabulary size: %d" % len(vocabulary))

        print("word vector of 'formed': ")
        print(word_vectors['formed'].tolist())

        #print("word vector of '0000': ")
        #print(word_vectors['0000'].tolist())
        #print(",".join(word_vectors['formed']))

        return word_vectors

    def add_unknown_words(self, word_vecs, vocabulary, min_df=1, d=300):
        for word in vocabulary:
            if word not in word_vecs and vocabulary[word] >= min_df:
                #print word
                word_vecs[word] = np.random.uniform(-0.25, 0.25, d)

    def save_model(self, filename="word2vec.model"):
        w2v_file = open(globals.data_path + filename, "wb")
        pickle.dump(self.word_vecs, w2v_file, protocol=2)
        w2v_file.close()

        self.save_model_words()

    def save_model_words(self, filename="word2vec.words"):
        w2vw_file = open(globals.data_path + filename, "wb")
        self.word_ids = {}
        for k, v in enumerate(self.word_vecs):
            self.word_ids[v] = k

        self.word_ids['_padding'] = len(self.word_ids)
        pickle.dump(self.word_ids, w2vw_file, protocol=2)
        w2vw_file.close()

    def load_model(self, filename="word2vec.model"):
        w2v_file = open(globals.data_path + filename, "rb")
        self.word_vecs = pickle.load(w2v_file)
        w2v_file.close()

        self.load_model_words()

    def load_model_words(self, filename="word2vec.words"):
        w2vw_file = open(globals.data_path + filename, "rb")
        self.word_ids = pickle.load(w2vw_file)
        w2vw_file.close()

    def get_word_vec(self, word):
        return self.word_vecs[word]

    def get_word_id(self, word):
        if not self.word_ids or word not in self.word_ids:
            raise ValueError("word_ids not initialized or '%s' not in it." % word)
        return self.word_ids[word]

    def get_words_similarity(self, word1, word2):
        #print("Cosine for words: %s and %s" % (word1, word2))
        if word1 not in self.word_vecs or word2 not in self.word_vecs:
            return 0.0
        return 1.0 - cosine(self.word_vecs[word1.lower()], self.word_vecs[word2.lower()])

    def get_indexes_vecs(self):
        if not self.word_ids or not self.word_vecs:
            raise ValueError("word_ids or word_vecs not initialized")

        # +1 is on the right, because in word_ids there is a special entry for padding that will be padding
        if len(self.word_ids) != len(self.word_vecs) + 1:
            raise ValueError("len of words_ids (%s) != len of word_vecs(%s)" % (len(self.word_ids),
                                                                                len(self.word_vecs)+1))

        weights = []
        counter = 0
        for t in sorted(self.word_ids.items(), key=operator.itemgetter(1)):
            #print("counter: %s, t[0]: %s, t[1]: %s" % (counter, t[0], t[1]))
            if t[1] != counter:
                raise ValueError("Ordering in word_ids is broken.")
            if t[0] == '_padding':
                #print ("t[0] = _padding for t[1]: %s" % t[1])
                break

            weights.append(self.word_vecs[t[0]])
            counter += 1

        # The last one is padding entry
        weights.append([0]*globals.dimension)

        return np.array(weights)