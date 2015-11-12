from gensim.models import Word2Vec
import numpy as np
from numpy.random import choice
import globals


class Word2VecModel:
    """
    Class for Word2Vec representation of a text
    """
    model = None

    def __init__(self, vocabulary, filename="../GoogleNews-vectors-negative300.bin"):
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

    def get_word_vec(self, word):
        return self.word_vecs[word]
