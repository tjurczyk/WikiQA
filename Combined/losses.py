import lasagne
import theano.tensor as T
import numpy as np
from lasagne.utils import one_hot

def define_lm_losses(lm_pred,lm_labels,hinge):
    # lm_labels = T.roll(input,1)
    # print 'lm_labels', theano.printing.debugprint(lm_labels, print_type=True)    

    # lm_labels = T.set_subtensor(lm_labels[0],0).flatten()
    # print 'lm_labels', theano.printing.debugprint(lm_labels, print_type=True)    

    if hinge:
        loss = lasagne.objectives.multiclass_hinge_loss
    else:
        loss = T.nnet.categorical_crossentropy
    lm_loss = T.mean(loss(lm_pred,lm_labels))
    return lm_loss

def define_candidate_losses(candidate_pred,labels,outputs,hinge,cost_sensitive,l1,l2):
    if hinge:
        loss = lasagne.objectives.binary_hinge_loss
    else:
        loss = lasagne.objectives.binary_crossentropy
    if cost_sensitive > 0.0:
        nonzeros = labels.nonzero()
        ONE = T.cast(T.ones_like(labels),'float32')
        CS = T.set_subtensor(ONE[nonzeros],np.float32(1+cost_sensitive))
        candidate_loss = loss(candidate_pred, labels)*CS
    else:
        candidate_loss = loss(candidate_pred, labels)
    if l1 > 0.0:
        candidate_l1_penalty = lasagne.regularization.regularize_layer_params_weighted(outputs[-1], lasagne.regularization.l1) * l1
        candidate_loss += candidate_l1_penalty

    if l2 > 0.0:
        candidate_l2_penalty = lasagne.regularization.regularize_layer_params_weighted(outputs[-1], lasagne.regularization.l2) * l2
        candidate_loss += candidate_l2_penalty

    candidate_loss = candidate_loss.mean()
    argmax_candidate = T.argmax(candidate_pred)
    labels_any = T.any(labels)
    return [candidate_pred, labels, candidate_loss, candidate_pred[argmax_candidate], labels[argmax_candidate], labels_any]

def grad_noise(rs, noise_eta, noise_decay, step):
    if noise_eta > 0:
        noise_std = noise_eta/(1+step)**noise_decay
        return rs.normal(size=(1,),avg=0.0,std=noise_std)[0]
    else:
        return 0
