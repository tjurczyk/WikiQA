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
    questions = {}

    questions['train'], voc, idf = load_questions_from_file('train', q_limit, vocabulary)
    questions['validate'], voc, idf = load_questions_from_file('validate', q_limit, vocabulary)
    questions['test'], voc, idf = load_questions_from_file('test', q_limit, vocabulary)
    logging.info("Questions loaded.")

    word2vec = w2vec(vocabulary, filename='../GoogleNews-vectors-negative300.bin')

    for question_set in ["train", "validate", "test"]:
        logger.info("Now working for set: %s" % question_set)

        counter = 1
        samples = []
        labels = []

        for q in questions[question_set]:
            # Create a question vector
            q_sampled = get_sentence_vector_list(q.question, word2vec)
            labels_q = []

            for idx, a in enumerate(q.answers):
                a_sampled = get_sentence_vector_list(a, word2vec)

                labels_q.append(int(idx in q.correct_answer))
                #print("\nAdding sample\n")
                samples.append(generate_sample(q_sampled, a_sampled))

            labels.append(labels_q)
            sys.stdout.write("\rQuestions parsed: %d/%s" % (counter, q_limit if q_limit > -1 else "all"))
            sys.stdout.flush()
            counter += 1

        sys.stdout.write("\n")
        sys.stdout.flush()
        logger.info("Input generated for %s" % question_set)

        (map_valid, error_index) = validate_feature_map(samples)
        if map_valid is False:
            raise ValueError("Feature map is not symmetrical (matrix) and so cannot be used. Indexes of error: " + str(error_index))

        data_set_path = globals.data_path + question_set + "."
        data_set_path += str(q_limit) if q_limit > -1 else "all"
        data_set_path += "/"


        if not exists(data_set_path):
            makedirs(data_set_path)

        logger.info("Dumping features and labels...")
        f_samples = open(data_set_path + globals.nn_features_file, "wb")
        pickle.dump(samples, f_samples, protocol=2)
        f_samples.close()

        f_labels = open(data_set_path + globals.nn_labels_file, "wb")
        pickle.dump(labels, f_labels, protocol=2)
        f_labels.close()

        print "samples size for: %s is: %d" % (question_set, len(samples))


def load_questions_from_file(mode, q_limit, vocabulary=None):
    questions = []
    n = 0
    idf = defaultdict(float)
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

            # If Word2Vec will use normalized numbers (0000),
            # update vocabulary with them
            if globals.normalize_numbers is True:
                    for i in get_normalized_numbers(words_set):
                        vocabulary[i] += 1

            # If new question entity
            if is_new_question or question is None:
                answers_count = 0
                question = Question(split_line[0], split_line[1])
            else:
                question.add_answer(split_line[1])

            # Add answer if found
            if split_line[2] == "1":
                question.add_correct_answer(answers_count)

            answers_count += 1

    # Calculate idf
    for k, v in vocabulary.items():
        idf[k] = np.log(n / vocabulary[k])

    return questions, vocabulary, idf


def validate_feature_map(f_map):
    """
    Validate if feature map is symmetrical.
    :param f_map: Map to validate
    :return: Boolean if map is valid.
    :rtype: tuple
    """

    # FIXME: Old code for multi-dimension
    # First dimension is samples
    # for idx, sample in enumerate(f_map):
    #     if len(sample) != depth:
    #         return False, (idx,)
    #
    #     # Second dimension is depths
    #     for idy, d in enumerate(sample):
    #         # 2*s_size, because s_size is for single sentence,
    #         # while the sample is q+a, so 2*s_size
    #         if len(d) != 2*s_size:
    #             return False, (idx, idy)
    #
    #         # Third dimension is words in sentence (should be s_size)
    #         for idz, s in enumerate(d):
    #             if len(s) != globals.dimension:
    #                 return False, (idx, idy, idz)

    for idx, sample in enumerate(f_map):
        if len(sample) != 1:
            return False, (idx,)

        # Second dimension is depths
        for idy, d in enumerate(sample):
            # 2*s_size, because s_size is for single sentence,
            # while the sample is q+a, so 2*s_size
            # FIXME: Added +1, because now we have an extra vector of 0s
            # between q and a
            if len(d) != 2*globals.s_size + 1:
                return False, (idx, idy)

            # Third dimension is words in sentence (should be s_size)
            for idz, s in enumerate(d):
                if len(s) != globals.dimension:
                    return False, (idx, idy, idz)

    return True, -1


def generate_sample(q_words, a_words):
    sample = []

    # This is old code for multiple dimensions
    #######################################################
    # dim1 is from first word to last-1
    #dim1 = pad_with_zeroes(q_words[:-1])
    #dim1.extend(pad_with_zeroes(a_words[:-1]))

    # dim2 is from second word to last
    #dim2 = pad_with_zeroes(q_words[1:])
    #dim2.extend(pad_with_zeroes(a_words[1:]))

    #if len(dim1) is not 2*s_size or len(dim2) is not 2*s_size:
    #    raise ValueError("Dim is not the right size. (dim1, dim2): " + str((len(dim1), len(dim2))) +
    #                     "\nQuestion: " + str(q_words) + "\nAnswer: " + str(a_words))

    #sample.append(dim1)
    #sample.append(dim2)
    #######################################################

    dim = pad_with_zeroes(q_words)
    # Add one extra list of zeroes (separator between q and a)
    dim.append([0]*globals.dimension)
    dim.extend(pad_with_zeroes(a_words))

    sample.append(dim)

    return sample


def pad_with_zeroes(s, length=globals.s_size):
    if s is None:
        ValueError("list of words for sentence 's' None.")

    # Add sentence vectors
    padded = copy.deepcopy(s)

    # Pad with zeroes with s_size
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
