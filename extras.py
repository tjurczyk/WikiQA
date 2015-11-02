from theano.tensor.signal import downsample
from theano.tensor.nnet.neighbours import images2neibs
from keras.layers.convolutional import *
import globals

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
                                        mode=globals.pooling_mode)
        return output


# class AveragePooling2D(MaxPooling2D):
#     def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True):
#         super(AveragePooling2D, self).__init__()
#         self.input = T.tensor4()
#         self.poolsize = tuple(poolsize)
#         self.stride = stride
#         self.ignore_border = ignore_border
#     def get_output(self, train):
#         X = self.get_input(train)
#         sums = images2neibs(X, neib_shape=(globals.s_size, 1)).sum(axis=-1)
#         counts = T.neq(images2neibs(X, neib_shape=(globals.s_size, 1)), 0).sum(axis=-1)
#         average = (sums/counts).reshape((X.shape[0], X.shape[1], 2, 1))
#         return average


class AveragePooling2DVariable(MaxPooling2D):
    def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True):
        super(AveragePooling2DVariable, self).__init__()
        self.input = T.tensor4()
        self.poolsize = tuple(poolsize)
        self.stride = stride
        self.ignore_border = ignore_border
    def get_output(self, train):
        X = self.get_input(train)
        # check if poolsize is symmetric. If not, step in neibs has to be set.

        if self.stride is not None:
            # rows_symmetrical = (X.shape[2] + 1)//2
            # step_val = (X.shape[2] - 1)//2
            sums = images2neibs(X, neib_shape=self.poolsize, neib_step=self.stride).sum(axis=-1)
            counts = T.neq(images2neibs(X, neib_shape=self.poolsize, neib_step=self.stride), 0).sum(axis=-1)
            average = (sums/counts).reshape((X.shape[0], X.shape[1], 2, 1))
        else:
            # rows_symmetrical = (X.shape[2])//2
            sums = images2neibs(X, neib_shape=self.poolsize).sum(axis=-1)
            counts = T.neq(images2neibs(X, neib_shape=self.poolsize), 0).sum(axis=-1)
            average = (sums/counts).reshape((X.shape[0], X.shape[1], 2, 1))
        return average