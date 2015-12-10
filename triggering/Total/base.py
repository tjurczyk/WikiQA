import numpy as np
import theano
from theano import tensor as T, function, printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers.base import Layer
from lasagne.random import get_rng
from sklearn.metrics import recall_score, precision_score
import random
import math
import cPickle as pickle
from collections import defaultdict
import os
import sys, getopt
from collections import OrderedDict
import time
from lasagne.updates import get_or_compute_grads
from lasagne import utils

def default_word():
    return 'UNK'
def default_vocab():
    return 2

seed = 1
random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))
# theano.config.compute_test_value = 'warn'
class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, repeats, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.repeats = repeats

    def get_output_shape_for(self, input_shape):
        return tuple([self.repeats*input_shape[0]] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        return theano.tensor.extra_ops.repeat(input, self.repeats,axis=0)

def dual_loss_adam(loss_or_grads_1, loss_or_grads_2, params_1, params_2, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    grads_1 = get_or_compute_grads(loss_or_grads_1, params_1)
    t_prev_1 = theano.shared(utils.floatX(0.))
    t_1 = t_prev_1 + 1
    a_t_1 = learning_rate*T.sqrt(1-beta2**t_1)/(1-beta1**t_1)

    grads_2 = get_or_compute_grads(loss_or_grads_2, params_2)
    t_prev_2 = theano.shared(utils.floatX(0.))
    t_2 = t_prev_2 + 1
    a_t_2 = learning_rate*T.sqrt(1-beta2**t_2)/(1-beta1**t_2)

    updates = OrderedDict()


    for param, g_t in zip(params_1, grads_1):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        step = a_t_1*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev_1] = t_1

    for param, g_t in zip(params_2, grads_2):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        step = a_t_2*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        if param in updates:
            updates[param] -= step
        else:
            updates[param] = param - step

    updates[t_prev_2] = t_2

    return updates

    for param, grad in zip(params_1, grads_1):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))
    for param, grad in zip(params_2, grads_2):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        if param in updates:
            updates[param] /= 2
            updates[param] -= (learning_rate * grad / T.sqrt(accu_new + epsilon))/2
        else:
            updates[param] = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))
    return updates

class Base(object):
    def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
        self.end_epoch = end_epoch
        self.start_epoch = 0
        self.macro_batch_size = macro_batch_size
        self.micro_batch_size = micro_batch_size
        self.best_qloss = 100
        self.best_qf1 = 0
        self.best_closs = 100
        self.best_cf1 = 0
        self.question_params = None
        self.candidate_params = None
        self.model = None
        self.name = name
        self.load = False
        self.save = False
        self.best_qbias = None
        self.best_cbias = None
        self.get_args()

        self.Q_MAX_LENGTH = 24
        self.C_MAX_LENGTH = 40
        self.MAX_N_CANDIDATES = 30

    def build_model(self):
        print("Building model and compiling functions...")


        questions_in = T.imatrix('questions')
        candidates_in = T.imatrix('candidates')
        q_masks_in = T.matrix('q_masks')
        c_masks_in = T.matrix('c_masks')
        c_labels_in = T.fvector('candidate_labels')
        i = T.iscalar()

        outputs = self.define_layers(questions_in,candidates_in,q_masks_in,c_masks_in)

        train_existence = T.clip(lasagne.layers.get_output(outputs[0], trainable=True), 1.0e-7, 1.0 - 1.0e-7)
        train_candidates = T.clip(lasagne.layers.get_output(outputs[1], trainable=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)

        test_existence = T.clip(lasagne.layers.get_output(outputs[0], deterministic=True), 1.0e-7, 1.0 - 1.0e-7)
        test_candidates = T.clip(lasagne.layers.get_output(outputs[1], deterministic=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        
        # theano.printing.pydotprint(test_existence, outfile="graph.png", var_with_name_simple=True)  

        self.question_params = lasagne.layers.get_all_params(outputs[0], trainable=True)
        self.candidate_params = lasagne.layers.get_all_params(outputs[1], trainable=True)

        existence_train_loss, existence_test_loss, candidate_train_loss, candidate_test_loss = self.define_losses(train_existence,test_existence,train_candidates,test_candidates,c_labels_in)
        
        # updates = lasagne.updates.adadelta(existence_train_loss, self.question_params)
        updates = lasagne.updates.adam(candidate_train_loss, self.candidate_params)
        updates = dual_loss_adam(candidate_train_loss, existence_train_loss, self.candidate_params, self.question_params, learning_rate=0.0005)
        # updates = lasagne.updates.rmsprop(candidate_train_loss, self.candidate_params)
        # updates = OrderedDict() # turn off optimizer to debug

        print(lasagne.layers.count_params(outputs[0]))

        qb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype=np.int32)
        cb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype=np.int32)
        qmb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype=np.float32)
        cmb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype=np.float32)
        lb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES), dtype=np.float32)

        self.questions_macro_batch = theano.shared(qb, borrow=True)
        self.candidates_macro_batch = theano.shared(cb, borrow=True)
        self.q_masks_macro_batch = theano.shared(qmb, borrow=True)
        self.c_masks_macro_batch = theano.shared(cmb, borrow=True)
        self.c_labels_macro_batch = theano.shared(lb, borrow=True)

        self.train_fn = theano.function([i], [existence_train_loss, candidate_train_loss, train_existence, train_candidates], updates=updates,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            c_labels_in: self.c_labels_macro_batch[i]
        })

        self.test_fn = theano.function([i], [existence_test_loss, candidate_test_loss, test_existence, test_candidates],
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            c_labels_in: self.c_labels_macro_batch[i]
        })

        self.train_debug_fn = theano.function([i], [existence_train_loss, candidate_train_loss, train_existence, train_candidates], updates=updates,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            c_labels_in: self.c_labels_macro_batch[i]
        })

    def define_losses(self,train_existence,test_existence,train_candidates,test_candidates,c_labels_in):
        existence_train_loss = (train_existence.mean()-T.cast(T.any(c_labels_in),'float32'))**2
        existence_test_loss = (test_existence.mean()-T.cast(T.any(c_labels_in),'float32'))**2
        candidate_train_loss = lasagne.objectives.binary_crossentropy(train_candidates, c_labels_in).mean()
        candidate_test_loss = lasagne.objectives.binary_crossentropy(test_candidates, c_labels_in).mean()
        return existence_train_loss,existence_test_loss, candidate_train_loss, candidate_test_loss

    def define_layers(self,questions_in,candidates_in,q_masks_in,c_masks_in):
        Q_N_HIDDEN = 100

        GRAD_CLIP = 10
        GRAD_STEPS = 40

        C_N_HIDDEN = 100
        SUBSPACE_DIM = 100

        question = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=questions_in)
        candidates = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=candidates_in)

        q_embedding = lasagne.layers.EmbeddingLayer(question, input_size=self.vocab_size, output_size=300, W=self.embedding)
        c_embedding = lasagne.layers.EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=300, W=q_embedding.W)

        q_embedding.params[q_embedding.W].remove('trainable')
        c_embedding.params[c_embedding.W].remove('trainable')

        # print('q_embedding',q_embedding.output_shape)
        # print('c_embedding',c_embedding.output_shape)

        q_embedding_reshape = lasagne.layers.ReshapeLayer(q_embedding,(1,self.Q_MAX_LENGTH,300))
        c_embedding_individual = lasagne.layers.ReshapeLayer(c_embedding,(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH,300))
        c_embedding_combined = lasagne.layers.ReshapeLayer(c_embedding,(1,self.MAX_N_CANDIDATES*self.C_MAX_LENGTH,300))
        print('RNN Input shape',q_embedding_reshape.output_shape)

        '''
        subspace
        # q_subspace = lasagne.layers.DenseLayer(q_embedding,self.Q_MAX_LENGTH*SUBSPACE_DIM, nonlinearity=lasagne.nonlinearities.tanh)
        # c_subspace = lasagne.layers.DenseLayer(c_embedding,self.C_MAX_LENGTH*SUBSPACE_DIM, nonlinearity=lasagne.nonlinearities.tanh)
        # print('c_subspace',c_subspace.output_shape)
        # q_subspace_reshape = lasagne.layers.ReshapeLayer(q_subspace,(1,self.Q_MAX_LENGTH,SUBSPACE_DIM))
        # c_subspace_individual = lasagne.layers.ReshapeLayer(c_subspace,(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH,SUBSPACE_DIM))
        # c_subspace_combined = lasagne.layers.ReshapeLayer(c_subspace,(1,self.MAX_N_CANDIDATES*self.C_MAX_LENGTH,SUBSPACE_DIM))
        '''
        print('qRep input shape', q_embedding_reshape.output_shape)
        qRep = lasagne.layers.GRULayer(q_embedding_reshape, Q_N_HIDDEN, grad_clipping=GRAD_CLIP, only_return_final=True)
        print('qRep output shape',qRep.output_shape)
        repeated_qRep = RepeatLayer(qRep,self.MAX_N_CANDIDATES)
        
        # lasagne.utils.create_param(lasagne.init.GlorotNormal,(C_N_HIDDEN,))
        hidden_update = lasagne.layers.Gate(W_cell=None,nonlinearity=lasagne.nonlinearities.tanh)
        resetgate = lasagne.layers.Gate(W_cell=None)
        updategate = lasagne.layers.Gate(W_cell=None)

        c_combined_Rep_f = lasagne.layers.GRULayer(c_embedding_combined, 2*C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_combined_Rep_b = lasagne.layers.GRULayer(c_embedding_combined, 2*C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_combined_Rep = lasagne.layers.ElemwiseSumLayer([c_combined_Rep_f,c_combined_Rep_b])

        # c_individual_Rep = lasagne.layers.GRULayer(c_embedding, C_N_HIDDEN, mask_input=c_mask, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        # cRep_backward = lasagne.layers.GRULayer(c_embedding, C_N_HIDDEN, mask_input=c_mask, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        # cRep = lasagne.layers.ElemwiseSumLayer([cRep_forward,cRep_backward])
        cRep_merge = lasagne.layers.ReshapeLayer(c_combined_Rep,(1,-1))
        # print('c_combined_Rep shape',c_combined_Rep.output_shape)
        qcRep_merge = lasagne.layers.ConcatLayer([cRep_merge,qRep],axis=1) # question rep is concatenated to candidate rep 
        # print('qcRep_merge shape',qcRep_merge.output_shape)

        existence_prediction = lasagne.layers.DenseLayer(qcRep_merge, num_units=50, nonlinearity=lasagne.nonlinearities.tanh)
        existence_prediction_out = lasagne.layers.DenseLayer(existence_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        
        # existence_prediction_out = lasagne.layers.DenseLayer(qcRep_merge, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print('existence_prediction_out output shape',existence_prediction_out.output_shape)

        print('c_individual_Rep input shape',c_embedding_individual.output_shape)
        c_individual_Rep_f = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_individual_Rep_b = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_individual_Rep = lasagne.layers.ElemwiseSumLayer([c_individual_Rep_f,c_individual_Rep_b])

        print('c_individual_Rep output shape',c_individual_Rep.output_shape)

        qcRep = lasagne.layers.ConcatLayer([repeated_qRep,c_individual_Rep]) # question is concatenated to each candidate
        repeated_existence_prediction = RepeatLayer(existence_prediction_out,self.MAX_N_CANDIDATES) # contain probability also concatenated
        feature_merge = lasagne.layers.ConcatLayer([qcRep,repeated_existence_prediction]) # KEY MOVE

        print('candidate_prediction input', feature_merge.output_shape)
        candidate_prediction = lasagne.layers.DenseLayer(feature_merge, num_units=50, nonlinearity=lasagne.nonlinearities.sigmoid)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        # candidate_prediction_out = lasagne.layers.DenseLayer(feature_merge, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print('candidate_prediction_out output', candidate_prediction_out.output_shape)
        return [existence_prediction_out, candidate_prediction_out]
        '''
        masks
        q_mask = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=q_masks_in)
        c_mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=c_masks_in)
        c_mask_combined = lasagne.layers.ReshapeLayer(c_mask,(1,self.MAX_N_CANDIDATES*self.C_MAX_LENGTH))
        print('q_mask shape',q_mask.output_shape)
        print('c_mask shape',c_mask.output_shape)
        
        # why runs?
        qRep = lasagne.layers.GRULayer(q_embedding_reshape, Q_N_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=c_mask_combined, only_return_final=True)
        # but this doesnt?
        qRep = lasagne.layers.GRULayer(q_embedding_reshape, Q_N_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=q_mask, only_return_final=True)
        
        c_combined_Rep = lasagne.layers.GRULayer(c_embedding_combined, C_N_HIDDEN, mask_input=c_mask_combined, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        '''

    def set_all(self, i, macro_batch_index):
        q = self.questions[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:1,:self.Q_MAX_LENGTH]
        c = self.candidates[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH]
        qm = self.q_masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:1,:self.C_MAX_LENGTH]
        cm = self.c_masks[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH]
        l = self.c_labels[i][macro_batch_index * self.macro_batch_size: (macro_batch_index + 1) * self.macro_batch_size,:self.MAX_N_CANDIDATES]
        assert q.shape == (self.macro_batch_size,1,self.Q_MAX_LENGTH), 'wrong question shape should be ' + str((self.macro_batch_size,1,self.Q_MAX_LENGTH)) + ' got ' + str(q.shape)
        assert c.shape == (self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), 'wrong candidate shape should be ' + str((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH)) + ' got ' + str(c.shape)
        assert qm.shape == (self.macro_batch_size,1,self.Q_MAX_LENGTH), 'wrong qmask shape should be ' + str((self.macro_batch_size,1,self.Q_MAX_LENGTH)) + ' got ' + str(qm.shape)
        assert cm.shape == (self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), 'wrong cmask shape should be ' + str((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH)) + ' got ' + str(cm.shape)
        assert l.shape == (self.macro_batch_size,self.MAX_N_CANDIDATES), 'wrong label shape should be ' + str((self.macro_batch_size,self.MAX_N_CANDIDATES)) + ' got ' + str(l.shape)
        self.questions_macro_batch.set_value(q, borrow=True)
        self.candidates_macro_batch.set_value(c, borrow=True)
        self.q_masks_macro_batch.set_value(qm, borrow=True)
        self.c_masks_macro_batch.set_value(cm, borrow=True)
        self.c_labels_macro_batch.set_value(l, borrow=True)
    
    def set_all_rest(self, i, remainder):
        q = np.lib.pad(self.questions[i][-remainder:,:1,:self.Q_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        c = np.lib.pad(self.candidates[i][-remainder:,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        qm = np.lib.pad(self.q_masks[i][-remainder:,:1,:self.Q_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        cm = np.lib.pad(self.c_masks[i][-remainder:,:self.MAX_N_CANDIDATES,:self.C_MAX_LENGTH], [(0,self.macro_batch_size-remainder),(0,0),(0,0)],'constant')
        l = np.lib.pad(self.c_labels[i][-remainder:,:self.MAX_N_CANDIDATES], [(0,self.macro_batch_size-remainder),(0,0)],'constant')
        assert q.shape == (self.macro_batch_size,1,self.Q_MAX_LENGTH), 'wrong question shape should be ' + str((self.macro_batch_size,1,self.Q_MAX_LENGTH)) + ' got ' + str(q.shape)
        assert c.shape == (self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), 'wrong candidate shape should be ' + str((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH)) + ' got ' + str(c.shape)
        assert qm.shape == (self.macro_batch_size,1,self.Q_MAX_LENGTH), 'wrong qmask shape should be ' + str((self.macro_batch_size,1,self.Q_MAX_LENGTH)) + ' got ' + str(qm.shape)
        assert cm.shape == (self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), 'wrong cmask shape should be ' + str((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH)) + ' got ' + str(cm.shape)
        assert l.shape == (self.macro_batch_size,self.MAX_N_CANDIDATES), 'wrong label shape should be ' + str((self.macro_batch_size,self.MAX_N_CANDIDATES)) + ' got ' + str(l.shape)

        self.questions_macro_batch.set_value(q, borrow=True)
        self.candidates_macro_batch.set_value(c, borrow=True)
        self.q_masks_macro_batch.set_value(qm, borrow=True)
        self.c_masks_macro_batch.set_value(cm, borrow=True)
        self.c_labels_macro_batch.set_value(l, borrow=True)

    def train_model(self):
        print('Starting Train')
        for epoch in xrange(self.start_epoch,self.end_epoch):
            start_time = time.time()
            self.epoch = epoch

            train_qloss = 0
            train_closs = 0
            dev_qloss = 0
            dev_closs = 0

            train_qpred = []
            train_cpred = []
            dev_qpred = []
            dev_cpred = []

            macro_batch_count = self.questions[0].shape[0] // self.macro_batch_size
            micro_batch_count = self.macro_batch_size // self.micro_batch_size
            for macro_batch_index in xrange(macro_batch_count):
                self.set_all(0, macro_batch_index)
                for micro_batch_index in xrange(micro_batch_count):
                    ''' print input looks OK
                    loss, question, candidates, prediction = self.train_debug_fn(micro_batch_index)
                    question_string = ''
                    for word in question[0]:
                        if word != 0:
                            question_string += self.vocab2word[word] + ' '
                    candidate_string = ''
                    for candidate in candidates:
                        for word in candidate:
                            if word != 0:
                                candidate_string += self.vocab2word[word] + ' '
                        candidate_string += '\n'
                    print(loss,question_string,candidate_string)
                    '''
                    # loss, question, candidates, prediction = self.train_debug_fn(micro_batch_index)
                    # print(loss)
                    t_ql, t_cl, t_qp, t_cp = self.train_fn(micro_batch_index)
                    train_qloss += t_ql
                    train_closs += t_cl
                    train_qpred.extend(t_qp)
                    train_cpred.extend(t_cp)
            if self.questions[0].shape[0] % self.macro_batch_size != 0:
                remainder = self.questions[0].shape[0]%self.macro_batch_size
                self.set_all_rest(0, remainder)
                for micro_batch_index in xrange(remainder):
                    t_ql, t_cl, t_qp, t_cp = self.train_fn(micro_batch_index)
                    train_qloss += t_ql
                    train_closs += t_cl
                    train_qpred.extend(t_qp)
                    train_cpred.extend(t_cp)

            macro_batch_count = self.questions[1].shape[0] // self.macro_batch_size
            micro_batch_count = self.macro_batch_size // self.micro_batch_size
            for macro_batch_index in xrange(macro_batch_count):
                self.set_all(1, macro_batch_index)
                for micro_batch_index in xrange(micro_batch_count):
                    d_ql, d_cl, d_qp, d_cp = self.test_fn(micro_batch_index)
                    dev_qloss += d_ql
                    dev_closs += d_cl
                    dev_qpred.extend(d_qp)
                    dev_cpred.extend(d_cp)
            if self.questions[1].shape[0] % self.macro_batch_size != 0:
                remainder = self.questions[1].shape[0]%self.macro_batch_size
                self.set_all_rest(1, remainder)
                for micro_batch_index in xrange(remainder):
                    d_ql, d_cl, d_qp, d_cp = self.test_fn(micro_batch_index)
                    dev_qloss += d_ql
                    dev_closs += d_cl
                    dev_qpred.extend(d_qp)
                    dev_cpred.extend(d_cp)

            train_qloss/=self.questions[0].shape[0]
            train_closs/=self.questions[0].shape[0]
            dev_qloss/=self.questions[1].shape[0]
            dev_closs/=self.questions[1].shape[0]

            train_question_labels = np.any(self.c_labels[0],axis=1,out=np.empty((self.c_labels[0].shape[0]),dtype=np.int32))
            train_candidate_labels = self.c_labels[0].flatten().astype(np.int32,copy=False)
            dev_question_labels = np.any(self.c_labels[1],axis=1,out=np.empty((self.c_labels[1].shape[0]),dtype=np.int32))
            dev_candidate_labels = self.c_labels[1].flatten().astype(np.int32,copy=False)

            dev_qpred, dev_qbias = self.find_best_threshold(np.array(dev_qpred),dev_question_labels,True,False)
            dev_cpred, dev_cbias = self.find_best_threshold(np.array(dev_cpred),dev_candidate_labels,False,False)

            train_qpred = np.rint(dev_qbias+train_qpred).astype(np.int32,copy=False)
            train_cpred = np.rint(dev_cbias+train_cpred).astype(np.int32,copy=False)
            train_qprec,train_qrec,train_qf1 = self.score(train_qpred,train_question_labels)
            train_cprec,train_crec,train_cf1 = self.score(train_cpred,train_candidate_labels)
            
            dev_qprec,dev_qrec,dev_qf1 = self.score(dev_qpred,dev_question_labels)
            dev_cprec,dev_crec,dev_cf1 = self.score(dev_cpred,dev_candidate_labels)
            end_time = time.time()
            print('Epoch Time', end_time-start_time)
            print('Question Epoch',self.epoch+1,'T-L:',train_qloss,'D-L:',dev_qloss,'D-P', dev_qprec, 'D-R', dev_qrec, 'T-F1:',train_qf1,'D-F1',dev_qf1, 'Best Loss', self.best_qloss, 'Best D-F1', self.best_qf1)
            print('Candidate Epoch',self.epoch+1,'T-L:',train_closs,'D-L:',dev_closs,'D-P', dev_cprec, 'D-R', dev_crec, 'T-F1:',train_cf1, 'D-F1',dev_cf1, 'Best Loss', self.best_closs, 'Best D-F1', self.best_cf1)
            # self.save_best(dev_qloss, dev_qf1, dev_closs, dev_cf1, dev_qbias, dev_cbias)

    def evaluate_model(self):
        print('Starting Test')
        test_qloss = 0
        test_closs = 0
        test_qpred = []
        test_cpred = []

        macro_batch_count = self.questions[2].shape[0] // self.macro_batch_size
        micro_batch_count = self.macro_batch_size // self.micro_batch_size
        for macro_batch_index in xrange(macro_batch_count):
            self.set_all(2, macro_batch_index)
            for micro_batch_index in xrange(micro_batch_count):
                t_ql, t_cl, t_qp, t_cp = self.test_fn(micro_batch_index)
                test_qloss += t_ql
                test_closs += t_cl
                test_qpred.extend(t_qp)
                test_cpred.extend(t_cp)
        if self.questions[2].shape[0] % self.macro_batch_size != 0:
            remainder = self.questions[2].shape[0]%self.macro_batch_size
            self.set_all_rest(2, remainder)
            for micro_batch_index in xrange(remainder):
                t_ql, t_cl, t_qp, t_cp = self.test_fn(micro_batch_index)
                test_qloss += t_qp
                test_closs += t_cp
                test_qpred.extend(t_qp)
                test_cpred.extend(t_cp)
        
        test_qpred, _ = self.find_best_threshold(np.array(test_qpred),True,True)
        test_cpred, _ = self.find_best_threshold(np.array(test_cpred),False,True)

        test_qprec,train_qrec,test_qf1 = self.score(test_qpred,np.any(self.c_labels[2]))
        test_cprec,train_crec,test_cf1 = self.score(test_cpred,self.c_labels[2].flatten())

        print('Test F1',test_qf1,'Test-P', test_qprec, 'Test-R', test_qrec)
        print('Test F1',test_cf1,'Test-P', test_cprec, 'Test-R', test_crec)
    
    def find_best_threshold(self,scores,labels,question,test):
        best_f1 = 0
        best_bias = 0
        if (test==True and question==True):
            best_pred = np.rint(scores+self.best_qbias)
            best_pred[best_pred > 1] = 1
            best_bias = self.qbest_bias
        if (test==True and question==False):
            best_pred = np.rint(scores+self.best_cbias)
            best_pred[best_pred > 1] = 1
            best_bias = self.cbest_bias
        else:
            for bias in np.arange(0,1,0.05):
                rb_pred = np.around(scores+bias)
                rb_prec = precision_score(labels,rb_pred)
                rb_rec = recall_score(labels,rb_pred)
                rb_f1 = 2*rb_prec*rb_rec/(rb_prec+rb_rec)
                if rb_f1 > best_f1:
                    best_f1 = rb_f1
                    best_pred = rb_pred
                    best_bias = bias
        return best_pred.astype(np.int32,copy=False),best_bias
    def load_data(self):
        self.questions = []
        self.candidates = []
        self.q_masks = []
        self.c_masks = []
        self.c_labels = []
        splits = ['train', 'validate', 'test']
        with open('../data/embedding.pickle', 'rb') as f:
            self.embedding,self.word2vocab,self.vocab2word,self.vocab_size = pickle.load(f)
        for split in splits:
            filepath = os.path.join('../data/', '%s.%s' % (split, 'npz'))
            data = np.load(filepath)
            self.questions.append(data['questions'].astype('int32'))
            self.candidates.append(data['candidates'].astype('int32'))
            self.q_masks.append(data['masks_q'].astype('float32'))
            self.c_masks.append(data['masks_c'].astype('float32'))
            self.c_labels.append(data['labels'].astype('float32'))
            data.close()


        bincount_q = np.bincount(self.questions[0].flatten())
        bincount_c = np.bincount(self.candidates[0].flatten())
        print('unknown in train questions',float(bincount_q[2])/(self.questions[0].size-bincount_q[0]))
        print('unknown in train candidates',float(bincount_c[2])/(self.candidates[0].size-bincount_c[0]))

        print(self.vocab2word[0])
        print(self.vocab2word[1])
        print(self.vocab2word[2])
        print(self.vocab2word[3])
        
        print(self.questions[0].shape)
        print(self.candidates[0].shape)
        print(self.q_masks[0].shape)
        print(self.c_masks[0].shape)
        print(self.c_labels[0].shape)

        ''' padding shape test
        print(self.c_labels[0][1 * 2000: (1 + 1) * 2000].shape)
        print(self.c_labels[0][-117:].shape)
        print(np.lib.pad(self.c_labels[0][-117:], [(0,2000-117),(0,0)],'constant').shape)
        '''
    def isInSubset():
        return False
    '''
    def createSubsetData():
        for i in xrange(3):
            for subset in PowerSet(range(30)):
                if len(subset) == 0:
                    self.candidates[i][subset]
                    self.c_masks[i][subset]
                    self.c_labels[i][subset]
                elif len(subset) == 1:
                
    '''
    def get_args(self):
        options, remainder = getopt.getopt(sys.argv[1:],'sl',['save','load'])
        for opt, arg in options:
            if opt in ('-s', '--save'):
                self.save = True
            elif opt in ('-l', '--load'):
                self.load = True

    def run(self):
        self.load_data()
        self.build_model()
        self.read_model_data()
        self.train_model()
        self.read_model_data()
        self.evaluate_model()
    def save_best(self,qloss,qf1,closs,cf1,qbias,cbias):
        if qloss < self.best_qloss:
            self.best_qloss = qloss
        if qf1 < self.best_qf1:
            self.best_qf1 = qf1
            self.best_qbias = qbias
        if closs < self.best_closs:
            self.write_model_data(qloss,qf1,closs,cf1)
            self.best_closs = closs
        if cf1 < self.best_cf1:
            self.best_cf1 = cf1
            self.best_cbias = cbias

    def read_model_data(self):
        if self.load == False:
            return
        print('Loading Model')
        modelFile = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
        if os.path.isfile(modelFile):
            with open(modelFile, 'rb') as f:
                self.start_epoch, self.best_qloss,self.best_qf1,self.best_closs,self.best_cf1, self.best_f1, params = pickle.load(f)
            self.end_epoch += self.start_epoch
            lasagne.layers.set_all_param_values(self.model,params)
    def write_model_data(self,qloss,qf1,closs,cf1):
        if self.save == False:
            return
        filename = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
        with open(filename, 'wb') as f:
            pickle.dump((self.epoch,qloss,qf1,closs,cf1, lasagne.layers.get_all_param_values(self.model)), f, protocol=-1)

    def score(self,pred,labels):
        prec = precision_score(labels,pred, average='binary')
        rec = recall_score(labels,pred, average='binary')
        f1 = 2*prec*rec/(prec+rec)
        return prec,rec,f1

def main():
    lstm = Base('bidir_lstm',2000,1,15)
    lstm.run()
if __name__ == '__main__':
    main()