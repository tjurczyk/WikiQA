from lasagne.layers import Layer, MergeLayer
import theano
import theano.tensor as T

class RepeatLayer(Layer):
    def __init__(self, incoming, repeats, axis=0, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.repeats = repeats
        self.axis=axis

    def get_output_shape_for(self, input_shape):
    	output_shape = list(input_shape)
    	output_shape[self.axis]*=self.repeats
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return T.repeat(input,self.repeats,axis=self.axis)


class MergeExpressionLayer(MergeLayer):
    def __init__(self, incoming, function, output_shape=None, **kwargs):
        super(MergeExpressionLayer, self).__init__(incoming, **kwargs)

        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.function = function

    def get_output_shape_for(self, input_shape):
        return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.function(*inputs)