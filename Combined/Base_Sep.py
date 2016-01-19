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
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data, print_input, create_lm_target
from Base import Base
from losses import define_lm_losses, define_candidate_losses
from Optimizers import gd_adam
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Base_Sep(Base):
    def __init__(self,name):
        self.start_epoch = 0
        self.end_epoch = 0
        self.epoch = 0

        self.start_p_epoch = 0
        self.end_p_epoch = 0
        self.p_epoch = 0

        self.macro_batch_size = 2200
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
    
    def define_layers(self, questions_in,candidates_in,q_masks_in,c_masks_in):
        pass

    def build_model(self):
        print('Building model')
        questions_in = T.imatrix('questions')
        candidates_in = T.imatrix('candidates')
        q_masks_in = T.matrix('q_masks')
        c_masks_in = T.matrix('c_masks')
        labels_in = T.ivector('candidates_labels')
        i = T.iscalar()
        e = T.fscalar()
        rs = RandomStreams(seed=1)

        outputs = self.define_layers(questions_in,candidates_in,q_masks_in,c_masks_in)
        if self.debug:
            print('defined layers')

        self.model = lasagne.layers.get_all_layers(outputs[:-1])

        train_candidate_pred = T.clip(lasagne.layers.get_output(outputs[0], trainable=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        test_candidate_pred = T.clip(lasagne.layers.get_output(outputs[0], deterministic=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)

        self.candidate_params = lasagne.layers.get_all_params(outputs[0], trainable=True)

        [train_candidate_pred, train_label, train_candidate_loss,
        train_top_candidate, train_top_candidate_label, train_labels_any] = define_candidate_losses(train_candidate_pred, labels_in, outputs, self.hinge, self.cost_sensitive, self.l1, self.l2)
        [test_candidate_pred, test_label, test_candidate_loss,
        test_top_candidate, test_top_candidate_label, test_labels_any] = define_candidate_losses(test_candidate_pred, labels_in, outputs, self.hinge, self.cost_sensitive, self.l1, self.l2)
        if self.debug:
            print('defined losses')

        print('learning rate {}'.format(self.learning_rate))
        print('{} parameters candidate'.format(lasagne.layers.count_params(outputs[0])))
        if self.debug:
            updates = OrderedDict()
        else:
            # updates = lasagne.updates.adam(train_candidate_loss, self.candidate_params, learning_rate=self.learning_rate)
            updates = gd_adam(train_candidate_loss, self.candidate_params, rs, e, self.noise_eta, self.noise_decay, learning_rate=self.learning_rate)
            
        qb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype='int32')
        cb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype='int32')
        qmb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype='float32')
        cmb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype='float32')
        lb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES), dtype='int32')

        self.questions_macro_batch = theano.shared(qb, borrow=True)
        self.candidates_macro_batch = theano.shared(cb, borrow=True)
        self.q_masks_macro_batch = theano.shared(qmb, borrow=True)
        self.c_masks_macro_batch = theano.shared(cmb, borrow=True)
        self.labels_macro_batch = theano.shared(lb, borrow=True)

        if self.debug:
            print('defined shared variables')

        self.train_fn = theano.function([i,e],
            [train_candidate_pred, train_label, train_candidate_loss,
            train_top_candidate, train_top_candidate_label, train_labels_any], updates=updates,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.test_fn = theano.function([i],
            [test_candidate_pred, test_label, test_candidate_loss,
            test_top_candidate, test_top_candidate_label, test_labels_any],
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        if self.debug:
            print('defined functions')

        if self.end_p_epoch  > 0:
            print('{} parameters lm'.format(lasagne.layers.count_params(outputs[1])))
            train_lm_pred = T.clip(lasagne.layers.get_output(outputs[1], trainable=True), 1.0e-7, 1.0 - 1.0e-7)
            test_lm_pred = T.clip(lasagne.layers.get_output(outputs[1], deterministic=True), 1.0e-7, 1.0 - 1.0e-7)
            # lm_target = T.concatenate([questions_in.repeat(self.MAX_N_CANDIDATES, axis=0),candidates_in], axis=1)

            self.lm_params = lasagne.layers.get_all_params(outputs[1], trainable=True)

            train_lm_loss = define_lm_losses(train_lm_pred,labels_in,self.hinge)
            test_lm_loss = define_lm_losses(test_lm_pred,labels_in,self.hinge)

            if self.debug:
                lm_updates = OrderedDict()
            else:
                lm_updates = lasagne.updates.adam(train_lm_loss, self.lm_params, learning_rate=self.learning_rate)

            self.train_lm_fn = theano.function([i,e], train_lm_loss, updates=lm_updates,
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
        print('Loading data')
        self.questions = []
        self.candidates = []
        self.q_masks = []
        self.c_masks = []
        self.labels = []
        self.episode_mask = []
        if self.dry:
            splits = ['validate','validate','validate']
        else:
            splits = ['train', 'validate', 'test']
        prefix = ''
        if (self.gn is True):
            prefix = 'GN_'
        with open('load/'+prefix+'embedding.pickle', 'rb') as f:
            self.embedding,self.vocab_size = pickle.load(f)
            self.EMBEDDING_DIM = self.embedding.shape[-1]
            print('Vector Dim',self.EMBEDDING_DIM)
        for split in splits:
            filepath = os.path.join('load/', '%s%s.%s' % (prefix,split, 'npz'))
            print('loading ' + filepath)
            data = np.load(filepath)
            self.questions.append(data['questions'].astype('int32'))
            self.candidates.append(data['candidates'].astype('int32'))
            self.q_masks.append(data['masks_q'].astype('float32'))
            self.c_masks.append(data['masks_c'].astype('float32'))
            self.labels.append(data['labels'].astype('int32'))
            self.episode_mask.append(np.any(self.c_masks[-1],axis=2))
            data.close()
        if self.check_input:
            self.print_input(vocab2word)
        print(self.episode_mask[-1].shape)
        print(self.labels[-1].shape)

    def set_all(self, i, macro_batch_index, pretrain=False):
        q = self.questions[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size, :1, :self.Q_MAX_LENGTH]
        c = self.candidates[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size, :self.MAX_N_CANDIDATES, :self.C_MAX_LENGTH]
        qm = self.q_masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size, :1, :self.C_MAX_LENGTH]
        cm = self.c_masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size, :self.MAX_N_CANDIDATES, :self.C_MAX_LENGTH]
        
        self.questions_macro_batch.set_value(q, borrow=True)
        self.candidates_macro_batch.set_value(c, borrow=True)
        self.q_masks_macro_batch.set_value(qm, borrow=True)
        self.c_masks_macro_batch.set_value(cm, borrow=True)
        if pretrain:
            l = create_lm_target(q,c)
        else:
            l = self.labels[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size, :self.MAX_N_CANDIDATES]
            # self.labels_macro_batch.set_value(np.eye(self.vocab_size)[l], borrow=True)
        self.labels_macro_batch.set_value(l, borrow=True)
        
    def set_all_rest(self, i, remainder, pretrain=False):
        q = np.lib.pad(self.questions[i][-remainder:, :1,:self.Q_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        c = np.lib.pad(self.candidates[i][-remainder:, :self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        qm = np.lib.pad(self.q_masks[i][-remainder:, :1,:self.Q_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        cm = np.lib.pad(self.c_masks[i][-remainder:, :self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        self.questions_macro_batch.set_value(q, borrow=True)
        self.candidates_macro_batch.set_value(c, borrow=True)
        self.q_masks_macro_batch.set_value(qm, borrow=True)
        self.c_masks_macro_batch.set_value(cm, borrow=True)
        if pretrain:
            l = create_lm_target(q,c)
        else:
            l = np.lib.pad(self.labels[i][-remainder:, :self.MAX_N_CANDIDATES], [(0,self.macro_batch_size-remainder),(0,0)],'constant')
            # self.labels_macro_batch.set_value(np.eye(self.vocab_size)[l], borrow=True)
        self.labels_macro_batch.set_value(l, borrow=True)