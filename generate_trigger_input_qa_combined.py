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
    # embedding = Word2Vec.load_word2vec_format('/home/mike/data/GoogleNews-vectors-negative300.bin', binary=True)
    # dim = 300
    embedding = Word2Vec.load_word2vec_format('/home/mike/data/wiki_nyt.skip.simplified.low.200.vectors.bin', binary=True)
    dim = 200
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
        combined = np.zeros((n_questions,max_n_candidates,maxlen_questions+1+maxlen_candidates+1),dtype=np.int32)
        masks = np.zeros((n_questions,max_n_candidates,maxlen_questions+1+maxlen_candidates+1),dtype=np.int8)
        only_candidate_mask = np.zeros(masks.shape)
        labels = np.zeros((n_questions,max_n_candidates),dtype=np.float32)

        for q_idx,q in enumerate(questions_split):
            q_len = add_q(q, word2vocab, combined, masks,only_candidate_mask, q_idx)
            for c_idx, c in enumerate(q.answers):
                add_c(c, word2vocab, combined, masks, only_candidate_mask, q_idx, c_idx, q_len)
                labels[q_idx,c_idx] = (int(c_idx in q.correct_answer))

        with open('./triggering/data/' + question_set + '_combined.npz', 'w') as f:
            np.savez(f,combined=combined, masks=masks, only_candidate_mask=only_candidate_mask, labels=labels)

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

def add_q(q, word2vocab, combined, masks,only_candidate_mask, q_idx):
    sentence = get_sentence_words(q.question)
    
    # do not mask the the question for the first candidate
    for i in xrange(len(sentence)+1):
        only_candidate_mask[q_idx,0,i] = 1

    for c_idx in xrange(len(q.answers)):
        for i,word in enumerate(sentence):
        # if word not in globals.p_marks:
        # if globals.normalize_numbers is True:
            #print("Extracting from word2vec word: %s" % re.sub("\d", "0", word))
            combined[q_idx,c_idx,i] = word2vocab[re.sub("\d", "0", word.lower())]
            masks[q_idx,c_idx,i] = 1
        masks[q_idx,c_idx,i+1] = 1
        combined[q_idx,c_idx,i+1] = 1
    return len(sentence)+1
        # else:
            # questions[q_idx,0,i] = word2vocab[word.lower()]

def add_c(s, word2vocab, combined, masks,only_candidate_mask, q_idx, c_idx, q_len):
    sentence = get_sentence_words(s)
    for i,word in enumerate(sentence):
        # if word not in globals.p_marks:
        # if globals.normalize_numbers is True:
            #print("Extracting from word2vec word: %s" % re.sub("\d", "0", word))
        combined[q_idx,c_idx,q_len+i] = word2vocab[re.sub("\d", "0", word.lower())]
        masks[q_idx,c_idx,q_len+i] = 1
        only_candidate_mask[q_idx,c_idx,q_len+i] = 1

    masks[q_idx,c_idx,q_len+i+1] = 1
    combined[q_idx,c_idx,q_len+i+1] = 1
    only_candidate_mask[q_idx,c_idx,q_len+i+1] = 1

        # else:
        #     candidates[q_idx,c_idx,i] = word2vocab[word.lower()]
if __name__ == '__main__':
    parse()
