from pprint import pprint
from math import log10
from nltk.stem import *
import globals


stemmer = PorterStemmer()


def decompose_questions(questions, stemming=False):
    samples = []

    for question in questions:
        samples.extend(decompose_question(question, stemming))
        #samples.append(decompose_question(question, stemming))

    return samples


def decompose_question(question_entity, stemming):
    """

    :param question_entity:
    :return:
    :rtype: list
    """

    samples = []
    idf = build_idf(question_entity, stemming)

    question = question_entity.question
    question_words = get_sentence_words_list(question, stemming)

    for idy, a in enumerate(question_entity.answers):
        answer = a
        answer_words = get_sentence_words_list(answer, stemming)

        wo = len(get_occurrence_items(question_words, answer_words))
        wo_idf = get_occurrence_items_idf(question_words, answer_words, idf)

        sample = [wo, wo_idf]
        samples.append(sample)

    return samples


def get_sentence_words_list(sentence, stemming=False):
    """
    Builds a list of words of a sentence with removed stop words and special characters
    :param sentence: String of sentence
    :return:         List of words in the sentence
    :rtype: list
    """

    word_list = []
    for i in sentence.split():
        word = ''.join(a for a in i if a.isalnum())
        if word != "" and word.lower() not in globals.stop_words and word.lower() not in globals.p_marks:
            if stemming:
                word_list.append(stemmer.stem(word.lower()))
            else:
                word_list.append(word.lower())
    return word_list


def get_occurrence_items(s1, s2):
    """
    Returns an array of co-occurring words in sentence1 and sentence2.
    :param s1: list of words in sentence 1
    :type s1: list
    :param s2: list of words in sentence 2
    :type s2: list
    :return: set of co-occurring words
    :rtype: set
    """

    #print("Trying to find an occurrence between: %s and %s." % (set(s1), set(s2)))
    return set(s1).intersection(set(s2))


def get_occurrence_items_idf(s1, s2, idf):
    intersection = set(s1).intersection(set(s2))
    score = 0.0

    for i in intersection:
        score += idf[i]

    return score


def build_idf(question_entity, stemming=False):
    idf = {}

    number_of_documents = 1 + len(question_entity.answers)
    dict_words = {}

    for answer in question_entity.answers:
        document_words = get_sentence_words_list(answer, stemming)
        for word in set(document_words):
            dict_words[word.lower()] = dict_words.get(word.lower(), 0) + 1

    for (k, v) in dict_words.items():
        idf[k] = log10(float(number_of_documents)/v)

    return idf