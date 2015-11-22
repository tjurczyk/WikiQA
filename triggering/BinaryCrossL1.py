from base import Triggering
from base import GaussianDropoutLayer
import lasagne
#Epoch 13 T-L: 0.0253055677923 D-L: 0.00283786583755 D-P 0.475 D-R 0.904761904762 D-F1 0.622950819672 Bias 0.45 Best Loss 0.00231086298571
#Test-L: 0.00148813964068 Test-P 0.391836734694 Test-R 0.79012345679 Test-F1 0.523874488404 Bias 0.45

class BinaryCrossL1(Triggering):
	def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
		super(BinaryCrossL1, self).__init__(name,macro_batch_size,micro_batch_size,end_epoch)

	def define_losses(self, prediction, test_prediction, l_in):
		loss = lasagne.objectives.binary_crossentropy(prediction, l_in).mean()
		test_loss = lasagne.objectives.binary_crossentropy(test_prediction, l_in).mean()
		l1_penalty = lasagne.regularization.regularize_network_params(flattened, l1) * 1e-4
		loss += l1_penalty
		return loss, test_loss

def main():
	lstm = BinaryCrossL1('bidir_context_convolutional_lstm_binarycrossl1',2000,200,50)
	lstm.run()
if __name__ == '__main__':
	main()