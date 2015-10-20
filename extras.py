from theano.tensor.signal import downsample
from keras.layers.convolutional import *

class AveragePooling2D(MaxPooling2D):
    def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True):
        super(AveragePooling2D, self).__init__()
        self.input = T.tensor4()
        self.poolsize = tuple(poolsize)
        self.stride = stride
        self.ignore_border = ignore_border
    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border,
                                        mode='average_exc_pad')
        return output