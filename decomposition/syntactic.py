from enum import Enum

import globals
import utils
from nlp_utils.dependency_structure import Sentence, Dependency_Structure


def decompose(q_words, a_words, q_dep, a_dep):
    """
    Extracts dependency features from question-answer pair

    :param q_words: list of words in a question
    :param a_words: list of words in an answer
    :rtype q_dep: dependency structure of a question
    :rtype a_dep: dependency structure of an answer
    """

    q_set, a_set = set(q_words), set(a_words)
    wo = q_set.intersection(a_set)

    for word in wo:
        print("Working on word: %s" % word)
        if word not in globals.stop_words:
            a = d_dispatch[globals.d_comparator](word, q_dep, a_dep)
            val = calculate_score(a, globals.d_metrics)
            print("returned list: %s, the calculated val: %f" % (a, val))


def string_word_similarity(word, q_dep, a_dep):
    """
    Calculate similarity based on word matching of parent-children
    :param word:
    :param q_dep:
    :param a_dep:
    :type word: basestring
    :type q_dep: Sentence
    :type a_dep: Sentence
    :return: A list of calculated similarities
    """

    q_word_index = q_dep.get_word_index(word)
    a_word_index = a_dep.get_word_index(word)

    similarity_values = []

    context_dict_q = q_dep.get_dependency_context(q_word_index)
    context_dict_a = a_dep.get_dependency_context(a_word_index)

    # Score for parent
    similarity_values.append(float(context_dict_q['head'].get_form() == context_dict_a['head'].get_form()))

    # Score for siblings
    siblings_score = 0.0
    for q_node in context_dict_q['siblings']:
        for a_node in context_dict_a['siblings']:
            siblings_score += float(q_node.form == a_node.form)
    similarity_values.append(siblings_score)

    # Score for children
    children_score = 0.0
    for q_node in context_dict_q['children']:
        for a_node in context_dict_a['children']:
            children_score += float(q_node.form == a_node.form)
    similarity_values.append(children_score)

    print("similar values: %s" % similarity_values)
    return similarity_values


class DCType(Enum):
    string_word, string_rel, string_word_rel, word_emb_word = range(4)


class DCMetrics(Enum):
    avg, max, sum = range(3)


def calculate_score(scores, dc_metrics):
    if dc_metrics == DCMetrics.avg:
        scores_sum = sum(scores)
        return float(scores_sum/len(scores))
    elif dc_metrics == DCMetrics.sum:
        return sum(scores)
    elif dc_metrics == DCMetrics.max:
        return max(scores)

d_dispatch = {
    DCType.string_word: string_word_similarity,
}

if __name__ == "__main__":
    ds = Dependency_Structure('../emory.txt.cnlp')
    #q_text = "Where Elephants live"
    #a_text = "Elephants are scattered throughout sub-Saharan Africa , and South and Southeast Asia ."
    q_text = "Broken Tomasz lost to the cinema yesterday and he watched an amazing movie."
    a_text = "Broken Tomasz lost his savings and he seemed to be very upset now."
    q = ds.sentences[0]
    a = ds.sentences[1]

    decompose(utils.get_sentence_words(q_text), utils.get_sentence_words(a_text), q, a)