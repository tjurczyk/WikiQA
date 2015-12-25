from collections import defaultdict
from gensim.models import Word2Vec
import cPickle as pickle
import argparse
import numpy as np
import sys
import copy
from os.path import exists
from os import makedirs
import os
from pprint import pprint
import logging
import re
from collections import OrderedDict
import itertools
from lasagne.regularization import l2, l1
import lasagne
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import theano
import theano.tensor as T
import random
import time
from utils import *
from Base import *

def default_word():
    return 'UNK'
def default_vocab():
    return 2
seed = 1
random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))

class Base_Sep(Base):
    def __init__(self,name):
        self.start_epoch = 0
        self.end_epoch = 0
        self.epoch = 0

        self.start_p_epoch = 0
        self.end_p_epoch = 0
        self.p_epoch = 0

        self.macro_batch_size = 3000
        self.micro_batch_size = 1
        self.model = None
        self.name = name

        self.learning_rate = 1e-3
        self.l1 = 1e-6
        self.l2 = 1e-6
        self.best_lm_loss = 100

        self.best_question_bias = 0
        self.best_question_f1 = 0

        self.best_candidate_loss = 100
        self.best_candidate_bias = 0
        self.best_candidate_f1 = 0

        self.MAX_N_CANDIDATES = 30
        self.Q_MAX_LENGTH = 24
        self.C_MAX_LENGTH = 40
        self.get_args()
    
    def define_layers(self, questions_in,candidates_in,q_masks_in,c_masks_in):
        pass

    def build_model(self):
        print('Building model')
        questions_in = T.imatrix('questions')
        candidates_in = T.imatrix('candidates')
        q_masks_in = T.matrix('q_masks')
        c_masks_in = T.matrix('c_masks')
        labels_in = T.fvector('candidates_labels')
        i = T.iscalar()

        outputs = self.define_layers(questions_in,candidates_in,q_masks_in,c_masks_in)

        self.model = lasagne.layers.get_all_layers(outputs)

        train_candidates_pred = T.clip(lasagne.layers.get_output(outputs[0], trainable=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        train_lm_pred = T.clip(lasagne.layers.get_output(outputs[1], trainable=True), 1.0e-7, 1.0 - 1.0e-7)

        test_candidates_pred = T.clip(lasagne.layers.get_output(outputs[0], deterministic=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        test_lm_pred = T.clip(lasagne.layers.get_output(outputs[1], deterministic=True), 1.0e-7, 1.0 - 1.0e-7)

        self.candidates_params = lasagne.layers.get_all_params(outputs[0], trainable=True)
        self.lm_params = lasagne.layers.get_all_params(outputs[1], trainable=True)

        [train_candidates_pred, train_candidates_label,
        train_candidates_loss, train_argmax_candidate] = self.define_candidate_losses(train_candidates_pred,labels_in, outputs)
        
        [test_candidates_pred, test_candidates_label,
        test_candidates_loss, test_argmax_candidate] = self.define_candidate_losses(test_candidates_pred,labels_in, outputs)

        train_lm_loss = self.define_lm_losses(train_lm_pred,candidates_in,outputs)
        test_lm_loss = self.define_lm_losses(test_lm_pred,candidates_in,outputs)

        print('learning rate {}'.format(self.learning_rate))
        print('{} parameters candidate'.format(lasagne.layers.count_params(outputs[0])))
        print('{} parameters lm'.format(lasagne.layers.count_params(outputs[1])))

        if self.debug:
            updates = OrderedDict()
            lm_updates = OrderedDict()
        else:
            updates = lasagne.updates.adam(train_candidates_loss, self.candidates_params, learning_rate=self.learning_rate)
            lm_updates = lasagne.updates.adam(train_lm_loss, self.lm_params, learning_rate=self.learning_rate)

        qb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype='int32')
        cb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype='int32')
        qmb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype=np.float32)
        cmb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype=np.float32)
        lb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES), dtype=np.float32)

        self.questions_macro_batch = theano.shared(qb, borrow=True)
        self.candidates_macro_batch = theano.shared(cb, borrow=True)
        self.q_masks_macro_batch = theano.shared(qmb, borrow=True)
        self.c_masks_macro_batch = theano.shared(cmb, borrow=True)
        self.labels_macro_batch = theano.shared(lb, borrow=True)

        self.train_fn = theano.function([i], [train_candidates_pred, train_candidates_label,
            train_candidates_loss,
            train_argmax_candidate], updates=updates,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.test_fn = theano.function([i], [test_candidates_pred, test_candidates_label,
            test_candidates_loss,
            test_argmax_candidate],
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.train_lm_fn = theano.function([i], train_lm_loss, updates=lm_updates,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.test_lm_fn = theano.function([i], test_lm_loss,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

    def load_data(self):
        self.questions = []
        self.candidates = []
        self.q_masks = []
        self.c_masks = []
        self.labels = []
        if self.dry:
            splits = ['validate','validate','validate']
        else:
            splits = ['train', 'validate', 'test']
        prefix = ''
        if (self.gn is True):
            prefix = 'GN_'
        with open('../../data/'+prefix+'embedding.pickle', 'rb') as f:
            self.embedding,word2vocab,vocab2word,self.vocab_size = pickle.load(f)
            self.EMBEDDING_DIM = self.embedding.shape[-1]
            print('Using '+prefix+' Vectors',self.EMBEDDING_DIM)            
        for split in splits:
            filepath = os.path.join('../../data/', '%s%s.%s' % (prefix,split, 'npz'))
            data = np.load(filepath)
            self.questions.append(data['questions'].astype('int32'))
            self.candidates.append(data['candidates'].astype('int32'))
            self.q_masks.append(data['masks_q'].astype('float32'))
            self.c_masks.append(data['masks_c'].astype('float32'))
            self.labels.append(data['labels'].astype('float32'))
            data.close()
        if self.check_input:
            self.print_input(vocab2word)

    def print_input(self, vocab2word):
        print('dev question shape {}'.format(self.questions[1].shape))
        print('dev question mask shape {}'.format(self.q_masks[1].shape))
        words = ''
        for i in xrange(self.questions[1].shape[0]):
            for j in xrange(self.questions[1].shape[2]):
                word = self.questions[1][i,0,j]
                mask = self.q_masks[1][i,0,j]
                if np.isclose(mask, 1.0):
                    words += vocab2word[word] + ' '
            words += '\n'
        print(words)
        
        print('dev candidates shape {}'.format(self.questions[1].shape))
        print('dev candidates mask shape {}'.format(self.q_masks[1].shape))
        words = ''
        for i in xrange(self.candidates[1].shape[0]):
            for j in xrange(self.candidates[1].shape[1]):
                for k in xrange(self.candidates[1].shape[2]):
                    word = self.candidates[1][i,j,k]
                    mask = self.c_masks[1][i,j,k]
                    if np.isclose(mask, 1.0):
                        words += vocab2word[word] + ' '
                words += '\n'
            words += '\n'
        print(words)
        # print('dev candidates shape {}'.format(self.candidates[1].shape))
        # for candidates in self.candidates[1]:
        #     words = ''
        #     for candidate in candidates:
        #         for word in candidate:
        #             words += vocab2word[word] + ' '
        #         words += '\n'
        #     print(words + '\n')
    def set_all(self, i, macro_batch_index):
        q = self.questions[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:1,:self.Q_MAX_LENGTH]
        c = self.candidates[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH]
        qm = self.q_masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:1,:self.C_MAX_LENGTH]
        cm = self.c_masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH]
        l = self.labels[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES]
        self.questions_macro_batch.set_value(q, borrow=True)
        self.candidates_macro_batch.set_value(c, borrow=True)
        self.q_masks_macro_batch.set_value(qm, borrow=True)
        self.c_masks_macro_batch.set_value(cm, borrow=True)
        self.labels_macro_batch.set_value(l, borrow=True)
    
    def set_all_rest(self, i, remainder):
        q = np.lib.pad(self.questions[i][-remainder:,:1,:self.Q_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        c = np.lib.pad(self.candidates[i][-remainder:,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        qm = np.lib.pad(self.q_masks[i][-remainder:,:1,:self.Q_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        cm = np.lib.pad(self.c_masks[i][-remainder:,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        l = np.lib.pad(self.labels[i][-remainder:,:self.MAX_N_CANDIDATES], [(0,self.macro_batch_size-remainder),(0,0)],'constant')
        self.questions_macro_batch.set_value(q, borrow=True)
        self.candidates_macro_batch.set_value(c, borrow=True)
        self.q_masks_macro_batch.set_value(qm, borrow=True)
        self.c_masks_macro_batch.set_value(cm, borrow=True)
        self.labels_macro_batch.set_value(l, borrow=True)


if __name__ == '__main__':
    lstm = Base_Sep('Base_Sep')
    lstm.run()