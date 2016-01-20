from enum import Enum

import globals
import utils
from nlp_utils.dependency_structure import Sentence, Dependency_Structure
from representation.wordrepresentation import Word2VecModel as word2vec
import sys


def decompose(q_words, a_words, q_dep, a_dep, comparator, emb_model=None):
    """
    Extracts dependency features from question-answer pair

    :param q_words: list of words in a question
    :param a_words: list of words in an answer
    :rtype q_dep: dependency structure of a question
    :rtype a_dep: dependency structure of an answer
    """

    q_set, a_set = set(q_words), set(a_words)
    wo = q_set.intersection(a_set)
    #print ("Start working on q_words: %s\na_words: %s\nq_dep_tokens: %s\na_dep_tokens: %s" % (q_words, a_words, q_dep.get_token_list(), a_dep.get_token_list()))

    # These three values are: parents_score, siblings_score and children_score
    sum_val = [0.0, 0.0, 0.0]

    for word in wo:
        #print("Working on word: %s" % word)
        if word not in globals.stop_words:
            val = d_dispatch[comparator](word, q_dep, a_dep, emb_model)
            #val = calculate_score(a, globals.d_metrics)
            #print("returned list: %s" % (val))
            sum_val = [i + j for i, j in zip(sum_val, val)]

    #print ("at the end, sum_val: %s" % sum_val)
    return sum_val


def string_word_similarity(word, q_dep, a_dep, emb_model):
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

    if q_word_index == -1 or a_word_index == -1:
        return [0.0, 0.0, 0.0]

    similarity_values = []

    context_dict_q = q_dep.get_dependency_context(q_word_index)
    context_dict_a = a_dep.get_dependency_context(a_word_index)

    # Score for parent
    similarity_values.append(float(get_word_typing(context_dict_q['head']) == get_word_typing(context_dict_a['head'])))

    # Score for siblings
    siblings_score = []
    for q_node in context_dict_q['siblings']:
        for a_node in context_dict_a['siblings']:
            siblings_score.append(float(get_word_typing(q_node) == get_word_typing(a_node)))
    #print ("siblings score: %s" % siblings_score)
    siblings_score_val = calculate_score(siblings_score, globals.d_metrics)
    similarity_values.append(siblings_score_val)

    # Score for children
    children_score = []
    for q_node in context_dict_q['children']:
        for a_node in context_dict_a['children']:
            children_score.append(float(get_word_typing(q_node) == get_word_typing(a_node)))
    children_score_val = calculate_score(children_score, globals.d_metrics)
    similarity_values.append(children_score_val)

    #print("similar values: %s" % similarity_values)
    return similarity_values


def embedding_word_similarity(word, q_dep, a_dep, emb_model):
    """
    Calculate similarity based on word matching of parent-children
    :param word:
    :param q_dep:
    :param a_dep:
    :type emb_model: Word2VecModel
    :type word: basestring
    :type q_dep: Sentence
    :type a_dep: Sentence
    :return: A list of calculated similarities
    """

    q_word_index = q_dep.get_word_index(word)
    a_word_index = a_dep.get_word_index(word)

    if q_word_index == -1 or a_word_index == -1:
        return [0.0, 0.0, 0.0]

    similarity_values = []

    context_dict_q = q_dep.get_dependency_context(q_word_index)
    context_dict_a = a_dep.get_dependency_context(a_word_index)

    # Score for parent
    if context_dict_q['head'].get_form() and context_dict_a['head'].get_form():
        #print ("word: %s, context q: %s, context a: %s" % (word, context_dict_q['head'].get_form(), context_dict_a['head'].get_form()))
        similarity_values.append(float(emb_model.get_words_similarity(context_dict_q['head'].get_form(),
                                                                     context_dict_a['head'].get_form())))
    else:
        similarity_values.append(0.0)

    # Score for siblings
    siblings_score = []
    for q_node in context_dict_q['siblings']:
        for a_node in context_dict_a['siblings']:
            if q_node.form and a_node.form:
                siblings_score.append(emb_model.get_words_similarity(q_node.form, a_node.form))
    #print ("siblings score: %s" % siblings_score)
    siblings_score_val = calculate_score(siblings_score, globals.d_metrics)
    similarity_values.append(siblings_score_val)

    # Score for children
    children_score = []
    for q_node in context_dict_q['children']:
        for a_node in context_dict_a['children']:
            if q_node.form and a_node.form:
                #print ("children, checking two nodes: %s and %s" % (q_node.form, a_node.form))
                children_score.append(emb_model.get_words_similarity(q_node.form, a_node.form))
    #print ("children score: %s" % children_score)
    children_score_val = calculate_score(children_score, globals.d_metrics)
    similarity_values.append(children_score_val)

    #print("similar values: %s" % similarity_values)
    return similarity_values


def get_word_typing(node):
    if globals.d_typing == "word_form":
        return node.get_form()
    elif globals.d_typing == "lemma":
        return node.get_lemma()
    else:
        raise ValueError("d_typing of '%s' is not supported." % globals.d_typing)


def calculate_score(scores, dc_metrics):
    if len(scores) == 0:
        return 0.0
    if dc_metrics == "avg":
        scores_sum = sum(scores)
        return float(scores_sum/len(scores))
    elif dc_metrics == "sum":
        return sum(scores)
    elif dc_metrics == "max":
        return max(scores)
    else:
        raise ValueError("dc_metrics not recognized: %s" % dc_metrics)

d_dispatch = {
    "string_word": string_word_similarity,
    "emb_word": embedding_word_similarity
}

if __name__ == "__main__":
    ds = Dependency_Structure('../emory.txt.cnlp')
    #q_text = "Where Elephants live"
    #a_text = "Elephants are scattered throughout sub-Saharan Africa , and South and Southeast Asia ."
    q_text = "Broken Tomasz lost to the cinema yesterday and he watched an amazing movie."
    a_text = "Broken Tomasz lost his savings and he seemed to be very upset now."
    q = ds.sentences[0]
    a = ds.sentences[1]

    try:
        word2vec = word2vec(None, filename=None)
        word2vec.load_model()
    except IOError:
        raise RuntimeError("Couldn't load word2vec")

    decompose(utils.get_sentence_words(q_text), utils.get_sentence_words(a_text), q, a, word2vec)