import sys
import cPickle as pickle
import copy
from os.path import exists
from os import makedirs
from pprint import pprint
from representation.wordrepresentation import Word2VecModel as w2vec
from representation.question import Question

import globals

# Option for cutting sentence vector list
# (so, if s_size is 40, but sentence is 44, first 40 tokens will be taken)
s_cut = True

# Settings for validation of file
# depth = 2

vec = w2vec()
data_set_path = None


def experiment():
    # List for samples (one sample is ibe
    samples = []
    labels = []

    questions = load_questions_from_file(globals.gen_mode, globals.q_limit)

    globals.logger.info("Generating input from questions...")
    counter = 1

    for q in questions:
        # Create a question vector
        q_repr = get_sentence_vector_list(q.question)
        labels_q = []

        for idx, a in enumerate(q.answers):
            a_repr = get_sentence_vector_list(a)

            labels_q.append(int(idx == q.correct_answer))
            samples.append(generate_sample(q_repr, a_repr))

        labels.append(labels_q)
        sys.stdout.write("\rQuestions parsed: %d/%s" % (counter, globals.q_limit if globals.q_limit > -1 else "all"))
        sys.stdout.flush()
        counter += 1
        #print q

    sys.stdout.write("\n")
    sys.stdout.flush()
    globals.logger.info("Input generated.")

    (map_valid, error_index) = validate_feature_map(samples)
    if map_valid is False:
        raise ValueError("Feature map is not symmetrical (matrix) and so cannot be used. "
                         "Indexes of error: " + str(error_index))

    globals.logger.info("Dumping features and labels...")
    f_samples = open(data_set_path + globals.nn_features_file, "wb")
    pickle.dump(samples, f_samples)
    f_samples.close()

    f_labels = open(data_set_path + globals.nn_labels_file, "wb")
    pickle.dump(labels, f_labels)
    f_labels.close()

    #pprint(samples)
    #pprint(labels)
    #pprint(samples)
    print "samples size = " + str(len(samples))


def load_questions_from_file(mode, q_limit):
    questions = []

    with open(globals.input_files.get(mode)) as f:
        # Forward one line (header)
        next(f)
        question_id = None
        question = None
        parsed_questions = 0
        answers_count = 0

        for line in f:
            split_line = line.rstrip().split('\t')

            # New question if current line's question id is different than previous
            is_new_question = question_id != split_line[0]

            if is_new_question:
                if question_id:
                    questions.append(question)
                    parsed_questions += 1
                    if -1 < q_limit == parsed_questions:
                        break

                question_id = split_line[0]
                answers_count = 0
                question = Question(split_line[0], split_line[3], split_line[1], [split_line[5], ])
            else:
                question.add_answer(split_line[5])

            if split_line[6] == "1":
                question.add_correct_answer(answers_count)

            answers_count += 1

    return questions


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


def get_sentence_vector_list(s):
    """
    :rtype :list
    """

    s_repr = []
    for j in s.split(" "):
        word = ''.join(e for e in j if e.isalnum())
        if word != "" and word.lower() not in globals.stop_words and word.lower() not in globals.p_marks:
            s_repr_word = vec.get_word_vec(word)
            if s_repr_word is not None:
                #print ("Getting wordvec of: %s, first elem: %s" % (word, s_repr_word[0]))
                s_repr.append(s_repr_word)

    if len(s_repr) > globals.s_size and not s_cut:
        raise ValueError("Size of sentence (q or a) exceeds s_size. (" + str(len(s_repr))
                         + ">" + str(globals.s_size) + ")")

    return s_repr[:globals.s_size]


def get_config():
    config = {"data_set_path": data_set_path}
    return config


if __name__ == "__main__":
    if globals.gen_mode is None:
        raise ValueError("Generation mode is None (run with -g)")

    data_set_path = globals.data_path \
                 + globals.gen_mode + "."
    data_set_path += str(globals.q_limit) if globals.q_limit > -1 else "all"
    data_set_path += "/"

    if not exists(data_set_path):
        makedirs(data_set_path)

    pprint(get_config())
    vec.load_model()
    experiment()
