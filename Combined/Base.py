from collections import defaultdict
from gensim.models import Word2Vec
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
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data, cosine, remove_model_data
from losses import define_lm_losses, define_candidate_losses
import cPickle as pickle

seed = 1
random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))

class Base(object):
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

        self.best_lm_loss = 100

        self.best_question_bias = 0
        self.best_question_f1 = 0

        self.best_candidate_loss = 100
        self.best_candidate_bias = 0
        self.best_candidate_f1 = 0

        self.MAX_N_CANDIDATES = 30
        self.MAX_LENGTH = 64

    def define_layers(self):
        pass

    def build_model(self):
        print('Building model')
        combined_in = T.imatrix()
        masks_in = T.matrix()
        ocm_in = T.matrix()
        labels_in = T.ivector()
        i = T.iscalar()

        outputs = self.define_layers(combined_in,masks_in,ocm_in)
        self.model = lasagne.layers.get_all_layers(outputs[:-1])

        train_candidate_pred = T.clip(lasagne.layers.get_output(outputs[0], trainable=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        train_lm_pred = T.clip(lasagne.layers.get_output(outputs[1], trainable=True), 1.0e-7, 1.0 - 1.0e-7)

        test_candidate_pred = T.clip(lasagne.layers.get_output(outputs[0], deterministic=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        test_lm_pred = T.clip(lasagne.layers.get_output(outputs[1], deterministic=True), 1.0e-7, 1.0 - 1.0e-7)

        self.candidate_params = lasagne.layers.get_all_params(outputs[0], trainable=True)
        self.lm_params = lasagne.layers.get_all_params(outputs[1], trainable=True)

        [train_candidates_pred, train_label, train_candidate_loss,
        train_top_candidate, train_top_candidate_label, train_labels_any] = define_candidate_losses(train_candidate_pred,labels_in, outputs)
        
        [test_candidates_pred, test_label, test_candidate_loss,
        test_top_candidate, test_top_candidate_label, test_labels_any] = define_candidate_losses(test_candidate_pred,labels_in, outputs)

        train_lm_loss = define_lm_losses(train_lm_pred,labels_in,outputs)
        test_lm_loss = define_lm_losses(test_lm_pred,labels_in,outputs)

        print('learning rate {}'.format(self.learning_rate))
        print('{} parameters candidate'.format(lasagne.layers.count_params(outputs[0])))
        print('{} parameters lm'.format(lasagne.layers.count_params(outputs[1])))

        if self.debug:
            updates = OrderedDict()
            lm_updates = OrderedDict()
        else:
            updates = lasagne.updates.adam(train_candidate_loss, self.candidate_params, learning_rate=self.learning_rate)
            lm_updates = lasagne.updates.adam(train_lm_loss, self.lm_params, learning_rate=self.learning_rate)

        combined = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.MAX_LENGTH), dtype=np.int32)
        masks = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.MAX_LENGTH), dtype=np.float32)
        ocm = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.MAX_LENGTH), dtype=np.float32)
        labels = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES), dtype=np.int32)

        self.combined_macro_batch = theano.shared(combined, borrow=True)
        self.masks_macro_batch = theano.shared(masks, borrow=True)
        self.ocm_macro_batch = theano.shared(ocm, borrow=True)
        self.labels_macro_batch = theano.shared(labels, borrow=True)

        self.train_fn = theano.function([i],
            [train_candidates_pred, train_label, train_candidate_loss,
            train_top_candidate, train_top_candidate_label, train_labels_any], updates=updates,
            givens={
            combined_in: self.combined_macro_batch[i],
            masks_in: self.masks_macro_batch[i],
            ocm_in: self.ocm_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.test_fn = theano.function([i],
            [test_candidates_pred, test_label, test_candidate_loss,
            test_top_candidate, test_top_candidate_label, test_labels_any],
            givens={
            combined_in: self.combined_macro_batch[i],
            masks_in: self.masks_macro_batch[i],
            ocm_in: self.ocm_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.train_lm_fn = theano.function([i], train_lm_loss, updates=lm_updates,
            givens={
            combined_in: self.combined_macro_batch[i],
            masks_in: self.masks_macro_batch[i],
            ocm_in: self.ocm_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

        self.test_lm_fn = theano.function([i], test_lm_loss,
            givens={
            combined_in: self.combined_macro_batch[i],
            masks_in: self.masks_macro_batch[i],
            ocm_in: self.ocm_macro_batch[i],
            labels_in: self.labels_macro_batch[i]
        })

    def get_i(self,split):
        if split == 'train':
            return 0
        elif split == 'dev':
            return 1
        elif split == 'test':
            return 2
        else:
            raise Exception('Invalid split')

    def pretrain(self,split):
        lm_loss = 0
        i = self.get_i(split)
        macro_batch_count = self.labels[i].shape[0] // self.macro_batch_size
        micro_batch_count = self.macro_batch_size // self.micro_batch_size
        remainder = self.labels[i].shape[0]%self.macro_batch_size

        if self.pbar:
            pbar = get_pbar(macro_batch_count*micro_batch_count+remainder)

        shuffled_macro_batches = range(macro_batch_count)
        random.shuffle(shuffled_macro_batches)
        for ma,macro_batch_index in enumerate(shuffled_macro_batches):
            self.set_all(i, macro_batch_index, pretrain=True)
            shuffled_micro_batches = range(micro_batch_count)
            random.shuffle(shuffled_micro_batches)
            for mi,micro_batch_index in enumerate(shuffled_micro_batches):
                if split == 'train':
                    lm_loss += self.train_lm_fn(micro_batch_index)
                else:
                    lm_loss += self.test_lm_fn(micro_batch_index)
                if self.pbar:
                    pbar.update(ma*micro_batch_count+mi)
        if remainder > 0:
            self.set_all_rest(i, remainder, pretrain=True)
            shuffled_remainder = range(remainder)
            random.shuffle(shuffled_remainder)
            for mi,remainder_index in enumerate(shuffled_remainder):
                if split == 'train':
                    lm_loss += self.train_lm_fn(remainder_index)
                else:
                    lm_loss += self.test_lm_fn(remainder_index)
                if self.pbar:
                    pbar.update(macro_batch_count*micro_batch_count+mi)
        if self.pbar:
            pbar.finish()

        lm_loss /= self.labels[i].shape[0]
        print(split + ' loss: {:.4f}'.format(float(lm_loss)))

        if split == 'dev' and lm_loss < self.best_lm_loss:
            self.best_lm_loss = lm_loss
            write_lm_data(self.save, self.best_lm_loss, self.name, self.p_epoch, self.best_lm_loss, self.model)
        
    def process(self,split):
        candidate_loss = 0
        candidate_preds = []
        candidate_label = []
        top_candidate = []
        top_candidate_label = []
        labels_any = []

        i = self.get_i(split)
        macro_batch_count = self.labels[i].shape[0] // self.macro_batch_size
        micro_batch_count = self.macro_batch_size // self.micro_batch_size
        remainder = self.labels[i].shape[0]%self.macro_batch_size
        
        if self.pbar:
            pbar = get_pbar(macro_batch_count*micro_batch_count+remainder)

        shuffled_macro_batches = range(macro_batch_count)
        random.shuffle(shuffled_macro_batches)
        for ma,macro_batch_index in enumerate(shuffled_macro_batches):
            self.set_all(i, macro_batch_index)
            shuffled_micro_batches = range(micro_batch_count)
            random.shuffle(shuffled_micro_batches)
            for mi,micro_batch_index in enumerate(shuffled_micro_batches):
                theano_func = self.train_fn
                if split != 'train':
                    theano_func = self.test_fn
                [micro_candidate_pred,micro_candidate,micro_candidate_loss,
                micro_top_candidate,micro_top_candidate_label,micro_labels_any] = theano_func(micro_batch_index)
                if self.pbar:
                    pbar.update(ma*micro_batch_count+mi)
                candidate_loss += micro_candidate_loss
                candidate_preds.append(micro_candidate_pred)
                candidate_label.append(micro_candidate)
                top_candidate.append(micro_top_candidate)
                top_candidate_label.append(micro_top_candidate_label)
                labels_any.append(micro_labels_any)
        if remainder != 0:
            self.set_all_rest(i, remainder)
            shuffled_remainder = range(remainder)
            random.shuffle(shuffled_remainder)
            for mi,remainder_index in enumerate(shuffled_remainder):
                theano_func = self.train_fn
                if split != 'train':
                    theano_func = self.test_fn
                [micro_candidate_pred,micro_candidate,micro_candidate_loss,
                micro_top_candidate,micro_top_candidate_label,micro_labels_any] = theano_func(remainder_index)

                if self.pbar:
                    pbar.update(macro_batch_count*micro_batch_count+mi)
                candidate_loss += micro_candidate_loss
                candidate_preds.append(micro_candidate_pred)
                candidate_label.append(micro_candidate)
                top_candidate.append(micro_top_candidate)
                top_candidate_label.append(micro_top_candidate_label)
                labels_any.append(micro_labels_any)

        candidate_loss/=self.labels[i].shape[0]
        if self.pbar:
            pbar.finish()

        candidate_preds = np.asarray(candidate_preds)
        candidate_label = np.asarray(candidate_label).flatten()
        top_candidate = np.asarray(top_candidate)
        top_candidate_label = np.rint(top_candidate_label).astype('int32')
        labels_any = np.asarray(labels_any).astype('bool')

        return self.evaluate(split,candidate_loss,candidate_preds,candidate_label,top_candidate,top_candidate_label,labels_any)

    def evaluate(self,split,
        candidate_loss,candidate_preds,candidate_label,top_candidate,top_candidate_label,labels_any):
        test = False
        if split is 'test':
            test = True
        candidate_preds_flatten = candidate_preds.flatten()
        corrected_candidate_pred, candidate_bias, candidate_prec,candidate_rec,candidate_f1 = self.find_best_threshold(candidate_preds_flatten,candidate_label,test,'binary')
        # ----------------------------- candidate ranking ----------------------------------------
        rank = np.argsort(candidate_preds_flatten)[::-1]
        correct = 0.0
        n_answers = len(candidate_label.nonzero()[0])
        prec_list = []
        rec_list = []
        f1_list = []
        for i,r in enumerate(rank,start=1):
            if candidate_label[r]==1:
                correct += 1.0
            prec = correct/i
            rec = correct/n_answers
            prec_list.append(prec)
            rec_list.append(rec)
            if prec+rec == 0:
                f1_list.append(0.)
            else:
                f1_list.append(100*2.*prec*rec/(prec+rec))
            if rec == 1:
                break
        avg_candidate_prec = np.asarray(prec_list).mean()
        print('max candidate IR F1 {:.2f}'.format(max(f1_list)))

        # prec,rec,thresholds = precision_recall_curve(candidate_label,candidate_preds_flatten)
        # f1_list = []
        # for p,r in zip(prec,rec):
            # f1_list.append(2.*p*r/(p+r))
        # argmax_f1 = np.argmax(f1_list)
        # ranked_candidate_f1 = f1_list[argmax_f1]
        # ranked_candidate_bias = thresholds[argmax_f1]
        # print('max candidate classification F1: {:.2f} bias: {:.2f}'.format(100*ranked_candidate_f1,ranked_candidate_bias))

        assert len(labels_any) == len(top_candidate_label), 'labels_any length: %r label length: %r' % (len(labels_any),len(top_candidate_label))
        assert len(top_candidate) == len(top_candidate_label), 'prediction length: %r label length: %r' % (len(top_candidate),len(top_candidate_label))
        corrected_question_pred, question_bias, question_prec, question_rec, question_f1 = \
        self.find_best_threshold_question(top_candidate,top_candidate_label,labels_any,test)

        self.print_scores(split,
            candidate_loss,candidate_prec,candidate_rec,candidate_f1,candidate_bias,
            question_prec,question_rec,question_f1,question_bias,
            avg_candidate_prec)
        if split == 'dev':
            if question_f1 > self.best_question_f1:
                self.save_best(
                    candidate_loss, candidate_f1, candidate_bias,
                    question_f1, question_bias)
        if test:
            sentence = self.name + ' lm_loss: {:.2f}\ndev candidate: {:.2f} question: {:.2f}\n test candidate: {:.2f} question: {:.2f}'.format(self.best_lm_loss, self.best_candidate_f1, self.best_question_f1, candidate_f1, question_f1)
            print(sentence)
            log(sentence, self.name)

        return question_f1

    def find_best_threshold(self,scores,labels,test,average='binary'):
        best_f1 = 0
        best_bias = 0
        best_pred = np.rint(scores).astype('int32')
        if test==True:
            best_pred = np.rint(scores+self.best_candidate_bias).astype('int32')
            best_pred[best_pred > 1] = 1
            best_bias = self.best_candidate_bias
        prec = precision_score(labels,best_pred,average=average)
        rec = recall_score(labels,best_pred,average=average)
        f1 = 0
        if prec + rec > 0:
            f1 = 2*prec*rec/(prec+rec)
        else:
            for bias in np.arange(0,1,0.05):
                pred = np.rint(scores+bias).astype('int32')
                prec = precision_score(labels,pred,average=average)
                rec = recall_score(labels,pred,average=average)
                f1 = 0
                if prec + rec > 0:
                    f1 = 2*prec*rec/(prec+rec)
                if f1 > best_f1:
                    best_f1 = f1
                    best_pred = pred
                    best_bias = bias
        return best_pred.astype(np.int32,copy=False),best_bias, prec,rec,f1

    def find_best_threshold_question(self,scores,labels,labels_any,test):
        best_prec = 0
        best_rec = 0
        best_f1 = 0 
        best_bias = 0
        best_pred = np.rint(scores).astype('int32')
        # if test==False:
        #     # self.best_question_bias = 0.3
        #     best_pred = np.rint(scores+self.best_question_bias).astype('int32')
        #     best_pred[best_pred > 1] = 1
        #     best_bias = self.best_question_bias
        #     best_prec, best_rec, best_f1 = self.score_question_alt(best_pred,labels,labels_any)
        # else:
        for bias in np.arange(0,1,0.01):
            pred = np.rint(scores+bias).astype('int32')
            prec, rec, f1 = self.score_question_alt(pred,labels,labels_any)
            # out = self.score_question(pred,labels,labels_any)
            # assert np.isclose(correctly_predicted_questions,out[3]), 'TP wrong %r %r' % (correctly_predicted_questions,out[3])
            # assert np.isclose(predicted_questions,out[4]), 'predicted questions wrong %r %r' % (predicted_questions,out[4])
            # assert np.isclose(all_questions_with_answers,out[5]), 'all question with answers wrong %r %r' % (all_questions_with_answers,out[5])
            # assert np.isclose(prec,out[0]), 'Precision wrong %r %r' % (prec,out[0])
            # assert np.isclose(rec,out[1]), 'Recall wrong %r %r' % (rec,out[1])
            # assert np.isclose(f1,out[2]), 'F1 wrong %r %r' % (f1,out[2])
            if f1 > best_f1:
                best_f1 = f1
                best_prec = prec
                best_rec = rec
                best_pred = pred
                best_bias = bias
        return best_pred.astype(np.int32,copy=False),best_bias,best_prec,best_rec,best_f1


    def score_question_alt(self,pred,labels,labels_any):
        pred = pred.astype('int32')
        labels = labels.astype('int32')
        labels_any = labels_any.astype('int32')
        predicted_questions = 0
        all_questions_with_answers = 0
        correctly_predicted_questions = 0
        for i in xrange(len(pred)):
            if labels_any[i]:
                all_questions_with_answers+=1
            if pred[i] == 1:
                predicted_questions += 1
                if labels[i] == 1:
                    correctly_predicted_questions += 1
                    # print('correctly predicted:',i)
                    # print('labels_any',labels_any[i])
                    # print('labels',labels[i])
        if correctly_predicted_questions == 0 or predicted_questions == 0:
            return 0.0, 0.0, 0.0
        precision= float(correctly_predicted_questions)/predicted_questions
        recall = float(correctly_predicted_questions)/all_questions_with_answers
        f1 = (2*precision*recall)/(precision+recall) if (precision > 0.0 and recall > 0.0) else 0.0
        # return precision,recall,f1,correctly_predicted_questions,predicted_questions,all_questions_with_answers
        return precision,recall,f1
    # BROKEN
    def score_question(self,predictions,labels,labels_any):
        TP = 0.
        candidate_FP = 0.
        FP = 0.
        FN = 0.
        TN = 0.
        pred_arr = []
        label_arr = []
        predictions = predictions.astype('int32')
        labels = labels.astype('int32')
        labels_any = labels_any.astype('int32')
        for i,pred in enumerate(zip(predictions,labels)):
            prediction = pred[0]
            label = pred[1]
            if labels_any[i]: # there exists an answer
                if prediction == 1: # top prediction is positive
                    if label == 1: # top prediction is correct
                        TP += 1
                        # print('tp:',i)
                        # print('labels_any',labels_any[i])
                        # print('labels',label)
                    else:
                        candidate_FP += 1 # top prediction is incorrect
                else: # top prediction is negative
                    FN += 1
            else: # there does not exist an answer
                assert labels[i] == 0, 'labels any wrong'
                if prediction == 1:
                    FP += 1 # predict candidate is an answer when no answer exists
                else:
                    TN += 1
        if TP+FP == 0:
            prec = 0.
        else:
            prec = TP/(TP+FP+candidate_FP)
        if TP+FN == 0:
            rec = 0.
        else:
            rec = TP/(TP+FN+candidate_FP) # the denominator is all 
        
        if prec + rec > 0:
            f1 = 2.*prec*rec/(prec+rec)
        else:
            f1 = 0.

        return prec,rec,f1,TP,TP+FP+candidate_FP,TP+FN+candidate_FP

    def save_best(self,
        candidate_loss, candidate_f1, candidate_bias,
        question_f1, question_bias):
        self.best_candidate_f1 = candidate_f1
        self.best_question_f1 = question_f1
        self.best_candidate_loss = candidate_loss

        self.best_candidate_bias = candidate_bias
        self.best_question_bias = question_bias

        write_model_data(self.save, self.name, self.epoch, candidate_loss, candidate_f1, candidate_bias,
            question_f1, question_bias, self.model)

    def print_scores(self,split,
        candidate_loss,candidate_prec,candidate_rec,candidate_f1,candidate_bias,
        question_prec,question_rec,question_f1,question_bias,
        avg_candidate_prec):
        c_string = '{} candidate L: {:.4f} P: {:.2f} R: {:.2f} F1: {:.2f} Bias: {:.2f}'.format(split, candidate_loss,100*candidate_prec,100*candidate_rec,100*candidate_f1, candidate_bias)
        q_string = '{} question P: {:.2f} R: {:.2f} F1: {:.2f} Bias: {:.2f}'.format(split, 100*question_prec,100*question_rec,100*question_f1, question_bias)
        ap_string = '{} average candidate precision: {:.2f}'.format(split,100*avg_candidate_prec)
        print(c_string)
        print(q_string)
        print(ap_string)

    def load_data(self):
        self.combined = []
        self.masks = []
        self.labels = []
        self.only_candidate_mask = []
        if self.dry:
            splits = ['validate','validate','validate']
        else:
            splits = ['train', 'validate', 'test']
        
        prefix = ''
        if (self.gn is True):
            prefix = 'GN_'
        with open('../../data/'+prefix+'embedding.pickle', 'rb') as f:
            self.embedding,self.vocab_size = pickle.load(f)
            self.EMBEDDING_DIM = self.embedding.shape[-1]
            print('Using '+prefix+' Vectors',self.EMBEDDING_DIM)            
        
        for split in splits:
            filepath = os.path.join('../../data/', '%s%s_combined.%s' % (prefix,split, 'npz'))
            data = np.load(filepath)
            self.combined.append(data['combined'].astype('int32'))
            self.masks.append(data['masks'].astype('float32'))
            self.only_candidate_mask.append(data['only_candidate_mask'].astype('float32'))
            self.labels.append(data['labels'].astype('int32'))
            data.close()

        bincount_q = np.bincount(self.combined[0].flatten())
        print('unknown train words',float(bincount_q[2])/(self.combined[0].size-bincount_q[0]))
        bincount_q = np.bincount(self.combined[0].flatten())
        print('unknown dev words',float(bincount_q[2])/(self.combined[1].size-bincount_q[1]))
        bincount_q = np.bincount(self.combined[0].flatten())
        print('unknown test words',float(bincount_q[2])/(self.combined[2].size-bincount_q[2]))
        
        if self.check_input:
            print_input()

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
        c = self.combined[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.MAX_LENGTH]
        m = self.masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.MAX_LENGTH]
        ocm = self.only_candidate_mask[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.MAX_LENGTH]
        l = self.labels[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES]
        self.combined_macro_batch.set_value(c, borrow=True)
        self.masks_macro_batch.set_value(m, borrow=True)
        self.ocm_macro_batch.set_value(m, borrow=True)
        self.labels_macro_batch.set_value(l, borrow=True)

    def set_all_rest(self, i, remainder):
        c = np.lib.pad(self.combined[i][-remainder:,:self.MAX_N_CANDIDATES,:self.MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        m = np.lib.pad(self.masks[i][-remainder:,:self.MAX_N_CANDIDATES,:self.MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        ocm = np.lib.pad(self.only_candidate_mask[i][-remainder:,:self.MAX_N_CANDIDATES,:self.MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        l = np.lib.pad(self.labels[i][-remainder:,:self.MAX_N_CANDIDATES], [(0,self.macro_batch_size-remainder),(0,0)],'constant')
        self.combined_macro_batch.set_value(c, borrow=True)
        self.masks_macro_batch.set_value(m, borrow=True)
        self.ocm_macro_batch.set_value(m, borrow=True)
        self.labels_macro_batch.set_value(l, borrow=True)

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr',type=float,help='learning rate',default=1e-3)
        parser.add_argument('--l1',type=float,help='l1 regularization',default=-1.0)
        parser.add_argument('--l2',type=float,help='l2 regularization',default=-1.0)
        parser.add_argument('--recurrent_layers',type=int,help='number of recurrent layers',default=1)
        parser.add_argument('--q_recurrent_layers',type=int,help='number of question recurrent layers',default=1)
        parser.add_argument('--c_recurrent_layers',type=int,help='number of candidate recurrent layers',default=1)
        parser.add_argument('--out_recurrent_units',type=int,help='dimensionality of out recurrent layer',default=0)
        parser.add_argument('--recurrent_units',type=int,help='dimensionality of recurrent layer',default=512)
        parser.add_argument('--attention_units',type=int,help='dimensionality of attention dense layer',default=512)
        parser.add_argument('--dense_units',type=int,help='dimensionality of dense layer',default=512)
        parser.add_argument('--attention_dropout',type=float,help='attention layer dropout',default=-1.)
        parser.add_argument('--out_recurrent_dropout',type=float,help='out recurrent layer dropout',default=-1.)
        parser.add_argument('--dense_dropout',type=float,help='dense layer dropout',default=-1.)
        parser.add_argument('--candidate_dropout',type=float,help='candidate input dropout',default=-1.)
        parser.add_argument('--question_dropout',type=float,help='question input dropout',default=-1.)
        parser.add_argument('--c_embedding_dropout',type=float,help='candidate embedding dropout',default=-1.)
        parser.add_argument('--q_embedding_dropout',type=float,help='question embedding dropout',default=-1.)
        parser.add_argument('--cost_sensitive',type=float,help='weigh positive class by its proportion')
        parser.add_argument('--trainable_embedding',action='store_true',help='sets trainable tag for embeddings')
        parser.add_argument('--embedding_attention',action='store_true',help='uses embedding instead of recurrent layers for attention')
        parser.add_argument('--slice_final',action='store_true',help='slice final recurrent layer (stacks)')
        parser.add_argument('--average_pool',action='store_true',help='mean pool final recurrent layer (stacks)')
        parser.add_argument('--max_pool',action='store_true',help='max pool final recurrent layer (stacks)')
        parser.add_argument('--average_pool_size',type=int,help='size of mean pooling layer',default=2)
        parser.add_argument('--max_pool_size',type=int,help='size of max pooling layer',default=2)
        parser.add_argument('--leakiness',type=float,help='rectifier leakiness, [0,1] -> [relu, linear]',default = 0.0)
        parser.add_argument('--bidirectional',action='store_true',help='use bidirectional recurrent layers')
        parser.add_argument('--gn',action='store_true',help='use google news vectors')
        parser.add_argument('--lstm',action='store_true',help='use lstm instead of GRU')

        parser.add_argument('--tmp',action='store_true',help='remove the saved model after evaluation')
        parser.add_argument('--hinge',action='store_true',help='use hinge loss instead of cross-entropy')
        parser.add_argument('--debug',action='store_true',help='sets debug')
        parser.add_argument('-d','--dry',action='store_true',help='trains and test on dev split to check for errors')
        parser.add_argument('-t','--train',type=int,help='min training epochs',default=0)
        parser.add_argument('-e','--evaluate',action='store_true',help='evaluate on test')
        parser.add_argument('-s','--save',action='store_true',help='save best model by dev')
        parser.add_argument('-l','--load',type=int,help='1 load pretrained, 2 load model',default=0)
        parser.add_argument('-p','--pretrain',type=int,help='final pretraining epoch',default=0)
        parser.add_argument('--pbar',action='store_true',help='use a progress bar, ugly file output')
        parser.add_argument('--rename',type=str,help='name modification')
        parser.add_argument('--check',action='store_true',help='input sanity check')

        args = parser.parse_args()
        # ---------------------------- spearmint hyperparams --------------------------
        self.learning_rate = args.lr
        self.l1 = args.l1
        self.l2 = args.l2
        
        self.q_recurrent_layers = args.q_recurrent_layers
        self.c_recurrent_layers = args.c_recurrent_layers
        self.recurrent_layers = args.recurrent_layers
        self.out_recurrent_units = args.out_recurrent_units
        self.recurrent_units = args.recurrent_units
        self.attention_units = args.attention_units
        self.dense_units = args.dense_units
        self.attention_dropout = args.attention_dropout
        self.question_dropout = args.question_dropout
        self.candidate_dropout = args.candidate_dropout
        self.q_embedding_dropout = args.q_embedding_dropout
        self.c_embedding_dropout = args.c_embedding_dropout
        self.out_recurrent_dropout = args.out_recurrent_dropout
        self.dense_dropout = args.dense_dropout
        self.cost_sensitive = args.cost_sensitive
        self.trainable_embedding = args.trainable_embedding
        self.embedding_attention = args.embedding_attention
        self.slice_final = args.slice_final
        self.average_pool = args.average_pool
        self.max_pool = args.max_pool
        self.average_pool_size = args.average_pool_size
        self.max_pool_size = args.max_pool_size
        self.leakiness = args.leakiness
        self.tmp = args.tmp
        if self.tmp:
            assert self.tmp == args.save == args.evaluate, 'you must save and evaluate your model if you want a temporary model'

        self.bidirectional = args.bidirectional
        self.gn = args.gn
        self.lstm = args.lstm


        # ----------------------------- additional options -----------------------------      
        self.debug = args.debug
        self.dry = args.dry
        self.end_epoch = args.train
        self.eval = args.evaluate
        self.save = args.save
        self.load = args.load
        self.end_p_epoch = args.pretrain
        self.hinge = args.hinge
        self.check_input = args.check
        self.pbar = args.pbar

        if args.rename != None:
            self.name += '-' + args.rename

        # log(str(args), self.name)

    def pretrain_train_model(self):
        print('Starting Pretrain')
        for p_epoch in xrange(self.start_p_epoch,self.end_p_epoch):
            start_time = time.time()
            self.pretrain('train')
            self.pretrain('dev')
            print('Pretrain epoch: {} {} sec\n'.format(p_epoch+1,int(time.time()-start_time)))
        
        print('Starting Train')
        epoch = self.start_epoch
        max_patience = 2
        patience = max_patience
        q_f1 = []
        while epoch < self.end_epoch+patience:
            start_time = time.time()
            epoch += 1
            self.process('train')
            q_f1.append(self.process('dev'))
            if epoch == 1:
                pass
            elif len(q_f1)-1 == np.argmax(q_f1):
                patience = max_patience
                print('New best question F1: {:.2f}'.format(q_f1[-1]))
            else:
                patience = max(0,patience-1)
            print('Epoch: {} {} sec\n'.format(epoch,int(time.time()-start_time)))

    def evaluate_model(self):
        print('Starting Test')
        start_time = time.time()
        candidate_loss = self.process('test')
        print('{} sec'.format(int(time.time()-start_time)))

    def run(self):
        self.load_data()
        self.build_model()
        
        if self.load == 1:
            self.start_p_epoch, self.best_lm_loss = read_lm_data(self.save, self.name, self.start_p_epoch, self.model)
        elif self.load == 2:
            self.start_epoch,self.best_candidate_loss,self.best_candidate_f1,self.best_candidate_bias,\
            self.best_question_f1,self.best_question_bias = read_model_data(self.load,self.name, self.model)
        
        if self.end_epoch-self.start_epoch > 0 or self.end_p_epoch-self.start_p_epoch > 0:
            self.pretrain_train_model()

        if self.load > 0:
            self.start_epoch,
            self.start_epoch,self.best_candidate_loss,self.best_candidate_f1,self.best_candidate_bias,\
            self.best_question_f1,self.best_question_bias = read_model_data(self.load,self.name, self.model)
        
        if self.eval:
            self.evaluate_model()
        
        remove_model_data(self.tmp, self.name)
        return self.best_candidate_loss

    def load_hyper(self, params):
        self.end_epoch = 2
        self.dry = False
        self.hinge = False
        self.debug = False
        self.load = 0
        self.eval = False
        self.check = False
        self.pretrain = 0
        self.lr = params['lr']
        self.l1 = -1
        self.l2 = -1
        self.q_recurrent_layers = params['q_recurrent_layers'][0]
        self.c_recurrent_layers = params['c_recurrent_layers'][0]
        self.recurrent_layers = 1
        self.recurrent_units = params['recurrent_units'][0]
        self.out_recurrent_units = params['out_recurrent_units'][0]
        self.attention_units = params['attention_units'][0]
        self.dense_units = params['dense_units'][0]
        self.attention_dropout = params['attention_dropout'][0]
        self.dense_dropout = params['dense_dropout'][0]
        self.out_recurrent_dropout = params['out_recurrent_dropout'][0]
        self.candidate_dropout = params['candidate_dropout'][0]
        self.question_dropout = params['question_dropout'][0]
        self.c_embedding_dropout = params['c_embedding_dropout'][0]
        self.q_embedding_dropout = params['q_embedding_dropout'][0]
        # self.leakiness = params['leakiness'][0]
        self.leakiness = 0
        self.embedding_attention = False
        self.trainable_embedding = False
        self.slice_final = True
        self.max_pool = False
        self.average_pool = False

        self.debug = True
        
        # if params['feature'] == ['slice_final']:
        #     self.slice_final = True
        # if params['feature'] == ['max_pool']:
        #     self.max_pool = True
        # if params['feature'] == ['average_pool']:
            # self.average_pool = True
        # self.max_pool_size = params['max_pool_size'][0]
        # self.average_pool_size = params['average_pool_size'][0]
        self.bidirectional = False
        self.gn = False
        self.lstm = False
        if params['lstm'] == ['lstm']:
            self.lstm = True
        self.cost_sensitive = params['cost_sensitive'][0]
        self.save = True
        self.check_input = False
        self.pbar = False

    def make_imports(self):
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
        from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data