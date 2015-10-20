from gensim.models import Word2Vec
from numpy.random import choice
import globals


class Word2VecModel:
    """
    Class for Word2Vec representation of a text
    """
    model = None

    def load_model(self, filename="../GoogleNews-vectors-negative300.bin"):
        globals.logger.info("Loading word2vec bin file...")
        self.model = Word2Vec.load_word2vec_format(filename, binary=True)
        globals.logger.info("Word2vec loaded.")

    def get_word_vec(self, word):
        #return choice(2, globals.dimension).tolist()
        return self.model[word] if word in self.model else None
