from __future__ import print_function
import numpy as np
import theano
from theano import tensor as T, function, printing
import lasagne
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from sklearn.preprocessing import normalize
from sklearn.metrics import recall_score, precision_score
import random
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
# seed = 1
# random.seed(seed)
# lasagne.random.set_rng(np.random.RandomState(seed))
Q_N_HIDDEN = 200
C_N_HIDDEN = 40
GRAD_CLIP = 10
GRAD_STEPS=40
# theano.config.compute_test_value = 'warn'
def build_LSTM():
	print("Building model and compiling functions...")
	q_in = T.ftensor3('questions') # batch x seqLen x features
	c_in = T.ftensor3('contexts') # batch x seqLen x features
	c_mask_in = T.fmatrix('q_mask')
	q_mask_in = T.fmatrix('c_mask')
	l_in = T.fvector('labels')
	
	print("Building network ...")
	q_l = lasagne.layers.InputLayer(shape=(None, 40, 300), input_var=q_in)
	q_mask_l = lasagne.layers.InputLayer(shape=(None, 40), input_var=q_mask_in)
	q_forward_1 = lasagne.layers.LSTMLayer(q_l, C_N_HIDDEN, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=q_mask_l)

	q_dense = lasagne.layers.DenseLayer(q_forward_1, 40)

	c_l = lasagne.layers.InputLayer(shape=(None, 1200, 300), input_var=c_in)
	c_mask_l = lasagne.layers.InputLayer(shape=(None, 1200), input_var=c_mask_in)
	c_forward_1 = lasagne.layers.LSTMLayer(c_l, C_N_HIDDEN, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=c_mask_l)

	a_dense = lasagne.layers.DenseLayer(c_forward_1, 260)

	l_concat = lasagne.layers.ConcatLayer([q_dense,a_dense])
	l_out = lasagne.layers.DenseLayer(l_concat, num_units=1, W = lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.sigmoid)

	# l_forward_2 = lasagne.layers.LSTMLayer(
		# l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP)

	#l_backward_1 = lasagne.layers.LSTMLayer(
	#    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP, backwards=True)

	# l_backward_2 = lasagne.layers.LSTMLayer(
			# l_backward_1, N_HIDDEN, grad_clipping=GRAD_CLIP, backwards=True)

	# l_backward_slice = lasagne.layers.SliceLayer(l_backward_1, 0, 1)

	# l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)

	# l_sum = lasagne.layers.ConcatLayer([l_forward_slice, l_backward_slice])
	# l_out = lasagne.layers.DenseLayer(l_backward_slice, num_units=2, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

	prediction = lasagne.layers.get_output(l_out)
	prediction = T.clip(T.flatten(prediction), 1.0e-7, 1.0 - 1.0e-7)
	
	test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
	test_prediction = T.clip(T.flatten(test_prediction), 1.0e-7, 1.0 - 1.0e-7)

	loss = lasagne.objectives.binary_crossentropy(prediction, l_in).mean()        
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction, l_in).mean()

	# layers = lasagne.layers.get_all_layers(l_out)
	params = lasagne.layers.get_all_params(l_out, trainable=True)
	updates = lasagne.updates.adadelta(loss, params)
	
	train_fn = theano.function([q_in, c_in, l_in, q_mask_in, c_mask_in], loss, updates=updates, allow_input_downcast=True)
	test_fn = theano.function([q_in, c_in, l_in, q_mask_in, c_mask_in], [test_loss, T.cast(T.round(test_prediction),'int32')], allow_input_downcast=True)
	recall_test_fn = theano.function([q_in, c_in, l_in, q_mask_in, c_mask_in], [test_loss, T.cast(T.round(test_prediction+0.2),'int32')], allow_input_downcast=True)
	# predict_fn = theano.function([q_in, c_in, l_in], [test_loss, test_prediction], allow_input_downcast=True)
	return train_fn, test_fn
	
def run_network(questions,contexts,labels,q_masks, c_masks):
	epochs = 15
	train_fn, test_fn = build_LSTM()

	q_train = questions[0]
	q_dev = questions[1]
	q_test = questions[2]

	q_mask_train = q_masks[0]
	q_mask_dev = q_masks[1]
	q_mask_test = q_masks[2]

	c_mask_train = c_masks[0]
	c_mask_dev = c_masks[1]
	c_mask_test = c_masks[2]

	c_train = contexts[0]
	c_dev = contexts[1]
	c_test = contexts[2]

	l_train = labels[0]
	l_dev = labels[1]
	l_test = labels[2]
	# Train

	print('questions shape',q_train.shape)
	print('contexts shape',c_train.shape)
	print('labels shape',l_train.shape)
	print('context mask shape',q_mask_train.shape)
	print('question mask shape',c_mask_train.shape)
	for epoch in range(epochs):
		train_loss = train_fn(q_train, c_train, l_train, q_mask_train, c_mask_train)
		valid_loss, valid_pred = test_fn(q_dev, c_dev, l_dev, q_mask_dev, c_mask_dev)
		r_valid_loss, r_valid_pred = recall_test_fn(q_dev, c_dev, l_dev, q_mask_dev, c_mask_dev)

		prec = precision_score(l_dev,valid_pred)
		rec = recall_score(l_dev,valid_pred)
		f1 = 2*prec*rec/(prec+rec)

		r_prec = precision_score(l_dev,r_valid_pred)
		r_rec = recall_score(l_dev,r_valid_pred)
		r_f1 = 2*prec*rec/(r_prec+r_rec)
		print('train:',train_loss,'valid:',valid_loss,'validation precision', prec, 'validation recall', rec, 'validation f1',f1)
		print('train:',train_loss,'valid_r:',r_valid_loss,'validation precision', r_prec, 'validation recall', r_rec, 'validation f1',r_f1)
	loss,pred = test_fn(q_test, c_test, l_test, q_mask_test, c_mask_test)
	auc = roc_auc_score(l_test, pred)
	acc = accuracy_score(l_test, pred)
	prec = precision_score(l_test, pred)
	rec = recall_score(l_test, pred)
	f1 = 2*prec*rec/(prec+rec)
	print('Average binary f1:', f1)
	print('Average binary precision:', prec)
	print('Average binary recall:', rec)
	print('Average binary auc:', auc)
	print('Average accuracy:', acc)
# def make_fig(X,Y):
#     mpl.rcParams['legend.fontsize'] = 10

#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     Y = np.loadtxt('Mice_Y.csv', dtype='int32')
#     X = []
#     for i in xrange(30):
#         color = 'r' if Y[i] ==1 else 'b'
#         a = np.loadtxt('miceX_' + str(i) + '.tsv').copy()
#         a.resize(7,2) # 26 is full
#         ax.plot(a[:,0],a[:,1],zs=np.arange(0,7),color=color,label='life of a mouse')
#     plt.show()
def main():
	questions = []
	contexts = []
	labels = []
	c_masks = []
	q_masks = []
	for question_set in ["train", "validate", "test"]:
		data = np.load(question_set + '.npz')
		questions.append(data['questions'])
		contexts.append(data['contexts'])
		labels.append(data['labels'])
		c_masks.append(data['c_masks'])
		q_masks.append(data['q_masks'])
		print(data['c_masks'][0],data['q_masks'][0])
		data.close()
	run_network(questions,contexts,labels,q_masks,c_masks)
	# make_fig(X,Y)
	
if __name__ == '__main__':
	main()