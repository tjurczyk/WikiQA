from pprint import pprint
import re
import globals
import numpy as np
import sys
from collections import defaultdict

import lexical
import syntactic
import utils


class SentenceExtractor():
    def __init__(self, features, word2vec):
        self.idf = None
        self.features = features
        self.word2vec = word2vec

    def decompose_questions(self, questions):
        if "wo_idf" in self.features and self.idf is None:
            raise RuntimeError("idf feature is in the list, but no idf_dictionary has been initialized. "
                               "Call 'build_idf()' method before")

        samples = []

        for question in questions:
            samples.extend(self.decompose_question(question))

        return samples

    def decompose_question(self, question_entity):
        samples = []

        for idy, a in enumerate(question_entity.answers):
            sample = []
            for f in self.features:
                if f == "wo":
                    sample.append(lexical.word_co_occurrence(question_entity.question, a))
                elif f == "wo_idf":
                    sample.append(lexical.word_co_occurrence_idf(question_entity.question, a, self.idf))
                elif f == "q_len":
                    sample.append(lexical.question_length(question_entity.question))
                elif f == "dependency":
                    for i in globals.lr_dep_features:
                        sample.extend(syntactic.decompose(utils.get_sentence_words(question_entity.question),
                                                          utils.get_sentence_words(a),
                                                          question_entity.question_dep, question_entity.answer_dep[idy],
                                                          i, self.word2vec))

            samples.append(sample)

        # print("samples are: %s" % samples)
        # sys.exit(0)
        return samples

    def build_idf(self, l_of_data_sets):
        self.idf = defaultdict(float)

        number_of_documents = 0.0

        for question_set in l_of_data_sets:
            for question in question_set:
                q_words = set(utils.get_sentence_words(question.question))
                n_of_answers = len(question.answers)

                for word in q_words:
                    if word in globals.stop_words:
                        continue
                    self.idf[word] += n_of_answers

                number_of_documents += n_of_answers

        for (k, v) in self.idf.items():
            self.idf[k] = np.log(float(number_of_documents)/v)


def get_normalized_numbers(l_words):
    norm_words = []

    for word in l_words:
        if contains_digits(word):
            norm_words.append(re.sub("\d", "0", word))

    return norm_words

_digits = re.compile('\d')


def contains_digits(d):
    return bool(_digits.search(d))
