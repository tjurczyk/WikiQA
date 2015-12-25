from progressbar import ProgressBar, Percentage, Bar, AdaptiveETA, Timer
import os
import cPickle as pickle
import lasagne

def get_pbar(maxval):
    return ProgressBar(widgets=[Percentage(),
    ' ', Bar(),
    ' ', AdaptiveETA(),
    ' (', Timer(), ')'],
    maxval=maxval).start()

def log_best(sentence, name):
    with open(name + '.log','w') as f:
        f.write(sentence)

def read_model_data(load, name, model):
    if load == False:
        return
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'params'))

    if os.path.isfile(modelFile):
        print('Loading Model')
        with open(modelFile, 'rb') as f:
            start_epoch,best_candidate_loss,best_candidate_f1,best_candidate_bias,\
            best_question_f1,best_question_bias,params = pickle.load(f)
        lasagne.layers.set_all_param_values(model,params)
        return start_epoch,best_candidate_loss,best_candidate_f1,best_candidate_bias,\
        best_question_f1,best_question_bias
    else:
        print('Model Not Found')
        return 0,100,0,0,0,0

def write_model_data(save, name, epoch, candidate_loss, candidate_f1, candidate_bias,
    question_f1, question_bias, model):
    if save == False:
        return
    print('Saving Model')
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'params'))
    with open(modelFile, 'wb') as f:
        pickle.dump((epoch,
            candidate_loss, candidate_f1, candidate_bias,
            question_f1, question_bias,
            lasagne.layers.get_all_param_values(model)), f, protocol=2)

def read_lm_data(save, name, start_p_epoch, model):
    if save == False:
        return
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'pretrain'))
    if os.path.isfile(modelFile):
        print('Loading LM')
        with open(modelFile, 'rb') as f:
            start_p_epoch,best_lm_loss,params = pickle.load(f)
        lasagne.layers.set_all_param_values(model,params)
    else:
        print('LM not found')
    return start_p_epoch, best_lm_loss

def write_lm_data(save, lm_loss, name, p_epoch, best_lm_loss, model):
    if save == False:
        return
    print('Saving LM')
    modelFile = os.path.join('./models/', '%s.%s' % (name, 'pretrain'))
    with open(modelFile, 'wb') as f:
        pickle.dump((p_epoch,best_lm_loss,lasagne.layers.get_all_param_values(model)), f, protocol=2)
