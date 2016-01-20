from nlp_utils.dependency_structure import Dependency_Structure
from representation.wordrepresentation import Word2VecModel as word2vec
from decomposition.utils import get_sentence_words
from decomposition.syntactic import decompose

try:
   word2vec = word2vec(None, filename=None)
   word2vec.load_model()
   print("Model loaded")
except IOError:
   raise RuntimeError("Couldn't load word2vec")

words = ['i', 'work', 'ones', '_padding']

for i in words:
    print ("word %s has id: %s" % (i, word2vec.get_word_id(i)))