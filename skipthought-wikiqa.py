from collections import defaultdict
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
import sys
import copy
from os.path import exists
from os import makedirs
from pprint import pprint
from representation.wordrepresentation import Word2VecModel as w2vec
from representation.question import Question
from decomposition.sentence import get_sentence_words, get_normalized_numbers
from collections import defaultdict
import logging
import re
import itertools
import globals
sys.path.insert(0, '/home/mike/coding/skip-thoughts')
import skipthoughts
model = skipthoughts.load_model()

q_limit = None
# def get_sentence_words(sentence):
#     return sentence.strip().lower().split()

def default_word():
    return 'UNK'
def default_vocab():
    return 2

def parse():
    print("loading embeddings")
    vocabulary = defaultdict(int)

    questions_f = {}
    questions_f['train'], _ = load_questions_from_file('train', q_limit, vocabulary)
    questions_f['validate'], _ = load_questions_from_file('validate', q_limit, vocabulary)
    questions_f['test'], _ = load_questions_from_file('test', q_limit, vocabulary)
    print('loading embeddings')
    embedding = Word2Vec.load_word2vec_format('/home/mike/data/GoogleNews-vectors-negative300.bin', binary=True)
    # embedding = Word2Vec.load_word2vec_format('./data/wiki_nyt.skip.word.100.vectors.bin', binary=True)

    dim = 300
    max_n_candidates, maxlen_questions, maxlen_candidates = find_maxes(questions_f)

    word2vocab = defaultdict(default_vocab)     
    vocab2word = defaultdict(default_word)
    word_set = vocabulary.keys()
    vocab_size = 3
    known = 0
    unknown = 0

    vocab2word[0] = ''
    vocab2word[1] = '<END>'
    vocab2word[2] = '<UNK>'

    word2vocab[''] = 0
    word2vocab['<END>'] = 1
    word2vocab['<UNK>'] = 2

    newEmbedding = [[0]*dim,[0]*dim,[0]*dim] # empty, end, unknown
    for word in word_set:
        word = re.sub("\d", "0", word)
        '''
        if word in embedding:
            newEmbedding.append(embedding[word])
        else:
            newEmbedding.append([0]*dim)
            unknown += 1
        word2vocab[word] = vocab_size
        vocab2word[vocab_size] = word
        '''
        if word in embedding:
            newEmbedding.append(embedding[word])
            word2vocab[word] = vocab_size
            vocab2word[vocab_size] = word
            vocab_size +=1
        else:
            word2vocab[word] = 2
            unknown += 1

    print vocab_size,unknown
    with open('./triggering/data/embedding.pickle', 'wb') as f:
        pickle.dump([np.asarray(newEmbedding,np.float32),word2vocab,vocab2word,vocab_size],f,protocol=2)

    for question_set in ["train", "validate", "test"]:
        print("Now working for set: %s" % question_set)
        questions_split = questions_f[question_set]
        n_questions = len(questions_split)
        # questions = np.zeros((n_questions,1,maxlen_questions),dtype=np.int32)
        # candidates = np.zeros((n_questions,max_n_candidates,maxlen_candidates),dtype=np.int32)
        # masks_q = np.zeros((n_questions,1,maxlen_questions),dtype=np.int8)
        # masks_c = np.zeros((n_questions,max_n_candidates,maxlen_candidates),dtype=np.int8)
        # labels = np.zeros((n_questions,max_n_candidates),dtype=np.float32)
        # st_q = np.zeros((n_questions,1,4800),dtype=np.float32)
        # st_c = np.zeros((n_questions,max_n_candidates,4800),dtype=np.float32)
        
        # st_c = []
        questions_list = []
        for q in questions_split:
            # answers_list = []
            # for answer in q.answers:
            #     answers_list.append(answer.strip().lower())
            # st_c.append(skipthoughts.encode(model, answers_list))
            questions_list.append(q.question.strip().lower())
        print(len(questions_list))
        st_q = skipthoughts.encode(model, questions_list)

        print st_q
        # print st_c

        # for q_idx,q in enumerate(questions_split):
        #     add_q(q.question, word2vocab, questions, masks_q, q_idx, st_q)
        #     for c_idx, c in enumerate(q.answers):
                # add_c(q.question, word2vocab, questions, masks_q, q_idx, c_idx)
                # add_c(c, word2vocab, candidates, masks_c, q_idx, c_idx, st_c)
                # labels[q_idx,c_idx] = (int(c_idx in q.correct_answer))

        with open('./triggering/data/' + question_set + 'skipthoughtsQ.npz', 'w') as f:
            np.savez(f,st_q=st_q)

    print('Max Sentence Length: ' + str(maxlen_questions))
    print('Recognized Words: ' + str(vocab_size-unknown))
    print('Unrecognized Words: ' + str(unknown))

def find_maxes(questions_f):
    max_n_candidates = 0
    maxlen_questions = 0
    maxlen_candidates = 0
    for question_set in ["train", "validate", "test"]:
        for q in questions_f[question_set]:
            if len(get_sentence_words(q.question)) > maxlen_questions:
                maxlen_questions = len(get_sentence_words(q.question))
                print(q.question)
                print('\n')
            if len(q.answers) > max_n_candidates:
                max_n_candidates = len(q.answers)
            for c in q.answers:
                if len(get_sentence_words(c)) > maxlen_candidates:
                    maxlen_candidates = len(get_sentence_words(c))
    print(max_n_candidates, maxlen_questions, maxlen_candidates)         
    return max_n_candidates, maxlen_questions, maxlen_candidates

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
            split_line = line.rstrip().lower().split('\t')

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

def add_q(s, word2vocab, questions, masks, idx, st_q):
    sentence = get_sentence_words(s)
    for i,word in enumerate(sentence):
        # if word not in globals.p_marks:
        if globals.normalize_numbers is True:
            #print("Extracting from word2vec word: %s" % re.sub("\d", "0", word))
            questions[idx,0,i] = word2vocab[re.sub("\d", "0", word.lower())]
            masks[idx,i] = 1
        else:
            questions[idx,0,i] = word2vocab[word.lower()]

def add_c(s, word2vocab, candidates, masks, q_idx, c_idx, st_c):
    sentence = get_sentence_words(s)
    for i,word in enumerate(sentence):
        # if word not in globals.p_marks:
        if globals.normalize_numbers is True:
            #print("Extracting from word2vec word: %s" % re.sub("\d", "0", word))
            candidates[q_idx,c_idx,i] = word2vocab[re.sub("\d", "0", word.lower())]
            masks[q_idx,c_idx,i] = 1
        else:
            candidates[q_idx,c_idx,i] = word2vocab[word.lower()]
if __name__ == '__main__':
    parse()
