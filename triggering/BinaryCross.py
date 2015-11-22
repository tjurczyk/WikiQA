from base import Triggering
from base import GaussianDropoutLayer
import lasagne

#Epoch 5 T-L: 0.00298449685678 D-L: 0.00235673492238 D-P 0.453531598513 D-R 0.968253968254 D-F1 0.617721518987 Bias 0.35 Best Loss 0.00231039766538
#Test-L: 0.00106248957447 Test-P 0.402826855124 Test-R 0.938271604938 Test-F1 0.563658838072 Bias 0.35

class BinaryCross(Triggering):
	def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
		super(BinaryCross, self).__init__(name,macro_batch_size,micro_batch_size,end_epoch)

	def define_losses(self, prediction, test_prediction, l_in):
		loss = lasagne.objectives.binary_crossentropy(prediction, l_in).mean()
		test_loss = lasagne.objectives.binary_crossentropy(test_prediction, l_in).mean()
		return loss, test_loss

def main():
	lstm = BinaryCross('bidir_context_convolutional_lstm_binarycross',2000,200,50)
	lstm.run()
if __name__ == '__main__':
	main()