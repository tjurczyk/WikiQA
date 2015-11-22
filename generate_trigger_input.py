import sys
import cPickle as pickle
import copy
from os.path import exists
from os import makedirs
from pprint import pprint
from representation.wordrepresentation import Word2VecModel as w2vec
from representation.question import Question
from decomposition.sentence import get_sentence_words, get_normalized_numbers
from collections import defaultdict
import numpy as np
import logging
import re

import globals

logger = logging.getLogger("wikiqa.generation")

# Settings for validation of file
# depth = 2

q_limit = None


def experiment():

    # List for samples (one sample is ibe

    vocabulary = defaultdict(float)
    questions_f = {}

    questions_f['train'], voc = load_questions_from_file('train', q_limit, vocabulary)
    questions_f['validate'], voc = load_questions_from_file('validate', q_limit, vocabulary)
    questions_f['test'], voc = load_questions_from_file('test', q_limit, vocabulary)
    logging.info("Questions loaded.")

    word2vec = w2vec(vocabulary, filename='./GoogleNews-vectors-negative300.bin')
    for question_set in ["train", "validate", "test"]:
        questions = []
        contexts = []
        labels = []
        q_masks = []
        c_masks = []
        logger.info("Now working for set: %s" % question_set)
        counter = 1
        for q in questions_f[question_set]:
            q_mask = []
            c_mask = []
            q_repr = get_sentence_vector_list(q.question, word2vec)
            question = pad_with_zero_vectors(q_repr)
            q_mask.extend([1]*len(q_repr))
            q_mask.extend([0]*(globals.s_size-len(q_repr)))
            context = []
            for a in q.answers:
                s_repr = get_sentence_vector_list(a, word2vec)
                context.extend(s_repr)
                c_mask.extend([1]*len(s_repr))
            while len(context) < 1200:
                context.append([0]*globals.dimension)
                c_mask.append(0)
            if len(q.correct_answer) > 0:
                labels.append(1)
            else:
                labels.append(0)
            sys.stdout.write("\rQuestions parsed: %d/%s" % (counter, q_limit if q_limit > -1 else "all"))
            sys.stdout.flush()
            contexts.append(context)
            questions.append(question)
            q_masks.append(q_mask)
            c_masks.append(c_mask)
            counter += 1
        logger.info("Input generated for %s" % question_set)
        contexts = np.asarray(contexts,dtype=np.float32)
        questions = np.asarray(questions,dtype=np.float32)
        labels = np.asarray(labels,dtype=np.float32)
        q_masks = np.asarray(q_masks,dtype=np.float32)
        c_masks = np.asarray(c_masks,dtype=np.float32)
        with open('./triggering/' +question_set+ '.npz','w') as f:
            np.savez(f, questions=questions, contexts=contexts, labels=labels, c_masks=c_masks, q_masks=q_masks)
def load_questions_from_file(mode, q_limit, vocabulary=None):
    questions = []
    n = 0
    if vocabulary is None:
        vocabulary = defaultdict(float)
    else:
        print("Vocabulary passed")

    with open(globals.input_files.get(mode)) as f:
        question_text = None
        question = None
        parsed_questions = 0
        answers_count = 0

        for line in f:
            split_line = line.rstrip().split('\t')

            # If new question (but not the first one)
            if question_text is not None and question_text != split_line[0]:
                is_new_question = True
                questions.append(question)
                parsed_questions += 1
            else:
                is_new_question = False

            # If there was a limit, break if reached
            if -1 < parsed_questions == q_limit:
                break

            question_text = split_line[0]

            # Number of samples/documents
            n += 1

            # Add to vocabulary
            words_set = set(get_sentence_words(split_line[0]))
            words_set.update(get_sentence_words(split_line[1]))
            for word in words_set:
                vocabulary[word] += 1
            if globals.normalize_numbers is True:
                    for i in get_normalized_numbers(words_set):
                        vocabulary[i] += 1
            if is_new_question or question is None:
                answers_count = 0
                question = Question(split_line[0], split_line[1])
            else:
                question.add_answer(split_line[1])
            if split_line[2] == "1":
                question.add_correct_answer(answers_count)
            answers_count += 1
    return questions, vocabulary

def generate_contextual_sample(q_words, a_words, context):
    dim = pad_with_zero_vectors(q_words)
    dim.append([0]*globals.dimension)
    dim.extend(pad_with_zero_vectors(a_words))
    dim.append([0]*globals.dimension)
    dim.append(context)
    # print len(dim)
    return [dim]
def pad_with_zero_vectors(s, length=globals.s_size):
    if s is None:
        ValueError("list of words for sentence 's' None.")
    if len(s) > globals.s_size:
        return s[:globals.s_size]
    # Add sentence vectors
    padded = copy.deepcopy(s)
    for _ in xrange(length - len(padded)):
        padded.append([0] * globals.dimension)
    return padded

def get_sentence_vector_list(s, word2vec):
    """
    :rtype :list
    """

    s_repr = []

    for word in get_sentence_words(s):
        if word not in globals.p_marks:
            if globals.normalize_numbers is True:
                #print("Extracting from word2vec word: %s" % re.sub("\d", "0", word))
                s_repr_word = word2vec.get_word_vec(re.sub("\d", "0", word))
            else:
                s_repr_word = word2vec.get_word_vec(word)

            s_repr.append(s_repr_word)

    return s_repr[:globals.s_size]


def get_config():
    config = {"q_limit": str(q_limit) if q_limit > -1 else "all"}
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if globals.gen_mode is None:
        raise ValueError("Generation mode is None (run with -g)")

    if not globals.options.q_limit:
        q_limit = -1
    else:
        q_limit = int(globals.options.q_limit)

    # data_set_path = globals.data_path \
    #              + globals.gen_mode + "."
    # data_set_path += str(q_limit) if q_limit > -1 else "all"
    # data_set_path += "/"

    print("Generation mode options:")
    p_order = ['q_limit']
    print(globals.get_printy_dict(get_config(), p_order))
    experiment()
