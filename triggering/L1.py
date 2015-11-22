from base import Triggering
from base import GaussianDropoutLayer
from lasagne.regularization import l2, l1
import lasagne

#Epoch 33 T-L: 0.0170154938678 D-L: 0.00104292722072 D-P 0.427118644068 D-R 1.0 D-F1 0.598574821853 Bias 0.5 Best Loss 0.000818867552078
#Epoch 34 T-L: 0.016515094039 D-L: 0.00108112935293 D-P 0.473251028807 D-R 0.912698412698 D-F1 0.623306233062 Bias 0.45 Best Loss 0.000818867552078
#Test-L: 0.000532411321809 Test-P 0.395833333333 Test-R 0.781893004115 Test-F1 0.525587828492 Bias 0.45

class L1(Triggering):
	def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
		super(L1, self).__init__(name,macro_batch_size,micro_batch_size,end_epoch)

	def define_losses(self, prediction, test_prediction, l_in):
		loss = lasagne.objectives.binary_crossentropy(prediction, l_in).mean()
		test_loss = lasagne.objectives.binary_crossentropy(test_prediction, l_in).mean()
		l1_penalty = lasagne.regularization.regularize_network_params(flattened, l1) * 1e-4
		loss += l1_penalty
		return loss, test_loss

def main():
	lstm = L1('bidir_context_convolutional_lstm_l1',2000,200,50)
	lstm.run()
if __name__ == '__main__':
	main()