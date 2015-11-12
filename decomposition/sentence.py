from pprint import pprint
import re
from math import log10
from collections import defaultdict
import globals
import numpy as np
import sys

idf = defaultdict(float)


def decompose_questions(questions):
    """
    Decompose set of questions.
    :param questions: Set of questions
    :type questions list
    :return: List of samples (one sample if question-sentence pair) of all questions
    :rtype: list
    """

    samples = []

    for question in questions:
        samples.extend(decompose_question(question))
        #samples.append(decompose_question(question))

    return samples


def decompose_question(question_entity):
    """
    Decompose a single question into set of samples.
    :param question_entity: Entity of a question
    :type question_entity: Question
    :return: List of samples (one sample is question-sentence pair)
    :rtype: list
    """

    samples = []

    question = question_entity.question
    question_words = get_sentence_words(question)

    idf_sum = 0.0
    q_length = len(question_words)

    wo_answers = float(get_occurrence_items_answers(question, question_entity.answers))#/q_length
    #print("wo_answers = %s" % wo_answers)
    #raw_input()

    for idy, a in enumerate(question_entity.answers):
        answer = a
        answer_words = get_sentence_words(answer)

        wo = get_occurrence_items(question_words, answer_words)
        wo_idf = get_occurrence_items_idf(question_words, answer_words)
        idf_sum += wo_idf

        sample = [wo, wo_idf, q_length,] #wo_answers]
        samples.append(sample)

    idf_average = float(idf_sum)/len(question_entity.answers)

    #for sample in samples:
    #    sample.extend([idf_average, sample[1]/idf_average if sample[1] != 0.0 else 0])

    #print("samples:")
    #pprint(samples)
    return samples


def get_occurrence_items_answers(q, answers):
    question_words = get_sentence_words(q)
    words = set()

    for a in answers:
        a_words = set(get_sentence_words(a))
        words.update(a_words.intersection(question_words))

    return len(words)


def get_sentence_words(sentence):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    return string.split()


def get_normalized_numbers(l_words):
    norm_words = []

    for word in l_words:
        if contains_digits(word):
            norm_words.append(re.sub("\d", "0", word))

    return norm_words

_digits = re.compile('\d')


def contains_digits(d):
    return bool(_digits.search(d))


def get_occurrence_items(s1, s2):
    """
    Returns an array of co-occurring words in sentence1 and sentence2.
    """

    q_set, a_set = set(s1), set(s2)
    count = 0.0

    for word in q_set:
        if word in a_set and word not in globals.stop_words:
            count += 1.0

    return count


def get_occurrence_items_idf(s1, s2):
    """
    Returns a normalized score of overlapping
    """

    q_set, a_set = set(s1), set(s2)
    score = 0.0

    for word in q_set:
        if word in a_set and word not in globals.stop_words:
            score += idf[word]

    return score


def build_idf(l_of_questions):
    """
    Build an idf dictionary for question set
    :param questions: Question set
    :type questions: list
    :return: idf dictionary
    :rtype: dict
    """

    #idf = defaultdict(float)

    number_of_documents = 0.0

    #pprint(questions)

    for question_set in l_of_questions:
        for question in question_set:
            q_words = set(get_sentence_words(question.question))
            n_of_answers = len(question.answers)

            for word in q_words:
                if word in globals.stop_words:
                    continue
                idf[word] += n_of_answers

            number_of_documents += n_of_answers

    # print("before dict: ")
    # pprint(idf)
    # print("n of documents: %d" % number_of_documents)

    for (k, v) in idf.items():
        idf[k] = np.log(float(number_of_documents)/v)

    # print("after dict: ")
    # pprint(idf)
    # print("n of documents: %d" % number_of_documents)

    #return idf