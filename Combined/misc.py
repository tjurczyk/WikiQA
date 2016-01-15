from progressbar import ProgressBar, Percentage, Bar, AdaptiveETA, Timer
import os
import cPickle as pickle
import lasagne
import numpy as np

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

def default_word():
    return 'UNK'

def default_vocab():
    return 2

def get_pbar(maxval):
    return ProgressBar(widgets=[Percentage(),
    ' ', Bar(),
    ' ', AdaptiveETA(),
    ' (', Timer(), ')'],
    maxval=maxval).start()

def save_plots(loss,candidate,question,name):
    with open('./plots/' + name + '.pickle','wb') as f:
        pickle.dump((loss,candidate,question),f,protocol=2)

def log(sentence, name):
    with open('./logs/' + name + '.log','a') as f:
        f.write(sentence)

def read_model_data(load, name, model):
    if load == False:
        return 0,100,0,0,0
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'params'))

    if os.path.isfile(modelFile):
        print('Loading Model')
        with open(modelFile, 'rb') as f:
            start_epoch, best_candidate_loss, best_candidate_f1,\
            best_question_f1, best_question_bias, params = pickle.load(f)
        lasagne.layers.set_all_param_values(model,params)
        return start_epoch,best_candidate_loss,best_candidate_f1,\
        best_question_f1,best_question_bias
    else:
        print('Model Not Found')
        return 0,100,0,0,0

def write_model_data(save, name, epoch, candidate_loss, candidate_f1,
    question_f1, question_bias, model):
    if save == False:
        return
    print('Saving Model')
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'params'))
    with open(modelFile, 'wb') as f:
        pickle.dump((epoch, candidate_loss, candidate_f1,
            question_f1, question_bias,
            lasagne.layers.get_all_param_values(model)), f, protocol=2)

def remove_model_data(tmp, name):
    if tmp == False:
        return
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'params'))
    os.remove(modelFile)
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'pretrain'))
    os.remove(modelFile)

def read_lm_data(save, name, start_p_epoch, model):
    if save == False:
        return 0,1000.0
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'pretrain'))
    if os.path.isfile(modelFile):
        print('Loading LM')
        with open(modelFile, 'rb') as f:
            start_p_epoch,best_lm_loss,params = pickle.load(f)
        lasagne.layers.set_all_param_values(model,params)
        return start_p_epoch, best_lm_loss
    else:
        print('LM not found')
        return 0,1000.0
def create_lm_target(q,c):
    q_roll = np.roll(q,1,axis=-1)
    q_roll = np.repeat(q_roll,c.shape[1],axis=1)
    c_roll = np.roll(c,1,axis=-1)
    c_roll[:,:,0] = q_roll[:,:,0]
    q_roll[:,:,0] = 0
    # EOS = np.ones((c_roll.shape[:2] + (1,)))
    return np.concatenate([q_roll,c_roll],axis=-1)

def write_lm_data(save, lm_loss, name, p_epoch, best_lm_loss, model):
    if save == False:
        return
    print('Saving LM')
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'pretrain'))
    with open(modelFile, 'wb') as f:
        pickle.dump((p_epoch,best_lm_loss,lasagne.layers.get_all_param_values(model)), f, protocol=2)

def slice_out(layer,index,axis):
    left = lasagne.layers.SliceLayer(layer, indices=slice(0, index), axis=axis)
    right = lasagne.layers.SliceLayer(layer, indices=slice(index+1, None), axis=axis)
    out = lasagne.layers.SliceLayer(layer, indices=slice(index, index+1), axis=axis)
    rest = lasagne.layers.concat([left,right],axis=axis)
    return out, rest
'''
Daniel Renshaw http://stackoverflow.com/questions/33523860/minimize-cosine-distance-theano
'''
def _squared_magnitude(x):
    return tt.sqr(x).sum(axis=-1)

def _magnitude(x):
    return tt.sqrt(tt.maximum(_squared_magnitude(x), numpy.finfo(x.dtype).tiny))

def cosine(x, y):
    return tt.clip((1 - (x * y).sum(axis=-1) / (_magnitude(x) * _magnitude(y))) / 2, 0, 1)

def euclidean(x, y):
    return _magnitude(x - y)

def squared_euclidean(x, y):
    return _squared_magnitude(x - y)