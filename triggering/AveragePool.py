from base import Triggering
from base import GaussianDropoutLayer
import lasagne

#Epoch 7 T-L: 0.00120304134245 D-L: 0.000816085631565 D-P 0.434027777778 D-R 0.992063492063 D-F1 0.6038647343 Bias 0.2 Best Loss 0.000816085631565
#Test-L: 0.000360034028941 Test-P 0.390522875817 Test-R 0.98353909465 Test-F1 0.559064327485 Bias 0.2

class AveragePool(Triggering):
	def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
		super(AveragePool, self).__init__(name,macro_batch_size,micro_batch_size,end_epoch)

	def define_layers(self, q_in,q_mask_in,c_in,c_mask_in):
		Q_N_HIDDEN = 40
		C_N_HIDDEN = 200
		GRAD_CLIP = 10
		GRAD_STEPS = 40

		q_l = lasagne.layers.InputLayer(shape=(None, 40, 300), input_var=q_in)
		q_conv = lasagne.layers.Conv1DLayer(q_l,40,3,pad='same')
		q_pool = lasagne.layers.Pool1DLayer(q_conv,3,mode='average_exc_pad')
		q_mask_l = lasagne.layers.InputLayer(shape=(None, 40), input_var=q_mask_in)
		q_resetgate=lasagne.layers.Gate(W_cell=None)
		q_updategate=lasagne.layers.Gate(W_cell=None)
		q_hidden_update=lasagne.layers.Gate(W_cell=None,nonlinearity=lasagne.nonlinearities.tanh)
		q_forward_1 = lasagne.layers.GRULayer(q_pool, Q_N_HIDDEN, resetgate=q_resetgate, updategate=q_updategate, hidden_update=q_hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=q_mask_l, only_return_final=True)

		c_l = lasagne.layers.InputLayer(shape=(None, 1200, 300), input_var=c_in)
		c_conv=lasagne.layers.Conv1DLayer(c_l,40,3,pad='same')
		c_pool = lasagne.layers.Pool1DLayer(c_conv,3,mode='average_exc_pad')
		c_mask_l = lasagne.layers.InputLayer(shape=(None, 1200), input_var=c_mask_in)
		c_resetgate=lasagne.layers.Gate(W_cell=None)
		c_updategate=lasagne.layers.Gate(W_cell=None)
		c_hidden_update=lasagne.layers.Gate(W_cell=None,nonlinearity=lasagne.nonlinearities.tanh)
		c_forward_1 = lasagne.layers.GRULayer(c_pool, C_N_HIDDEN, resetgate=c_resetgate, updategate=c_updategate, hidden_update=c_hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=c_mask_l, only_return_final=True)
		c_backward_1 = lasagne.layers.GRULayer(c_pool, C_N_HIDDEN, resetgate=c_resetgate, updategate=c_updategate, hidden_update=c_hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, mask_input=c_mask_l, only_return_final=True, backwards=True)
		c_sum = lasagne.layers.ElemwiseSumLayer([c_forward_1,c_backward_1])
		
		l_concat = lasagne.layers.ConcatLayer([q_forward_1,c_sum])
		l_out = lasagne.layers.DenseLayer(l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
		flattened = lasagne.layers.FlattenLayer(l_out, outdim=1)
		return flattened

def main():
	lstm = AveragePool('bidir_context_convolutional_lstm_averagepool',2000,200,50)
	lstm.run()
if __name__ == '__main__':
	main()