import globals
import utils


def word_co_occurrence(q, a):
    """
    Returns an array of co-occurring words in sentence1 and sentence2.
    """

    q_words = utils.get_sentence_words(q)
    a_words = utils.get_sentence_words(a)
    q_set, a_set = set(q_words), set(a_words)
    count = 0.0

    for word in q_set:
        if word in a_set and word not in globals.stop_words:
            count += 1.0

    return count


def word_co_occurrence_idf(q, a, idf):
    """
    Returns a normalized score of overlapping
    """

    q_words = utils.get_sentence_words(q)
    a_words = utils.get_sentence_words(a)
    q_set, a_set = set(q_words), set(a_words)
    score = 0.0

    for word in q_set:
        if word in a_set and word not in globals.stop_words:
            score += idf[word]

    return score


def question_length(q):
    question_words = utils.get_sentence_words(q)
    return len(question_words)