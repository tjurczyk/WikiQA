from Base import *

class Simple_2L(Base):
    def __init__(self,name):
        super(Simple_2L, self).__init__(name)

    def define_layers(self,combined_in,masks_in,ocm_in):
        # theano.tensor.extra_ops.cumsum nonzero == num words
        # check how many shared words, dot the bincounts, the get nonzeros / num words

        GRAD_CLIP = 15
        RECURRENT_N_HID = 498
        DENSE_N_HID = 512

        if self.lstm:
            RecurrentLayer = lasagne.layers.LSTMLayer
        else:
            RecurrentLayer = lasagne.layers.GRULayer

        mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.MAX_LENGTH), input_var=masks_in)
        ocm_mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.MAX_LENGTH), input_var=ocm_in)
        ocm_mask = lasagne.layers.ReshapeLayer(ocm_mask, shape=(1,-1))

        combined = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.MAX_LENGTH), input_var=combined_in)
        combined = lasagne.layers.WordDropoutLayer(combined, p=0.5)

        embedding = lasagne.layers.EmbeddingLayer(combined, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        embedding.params[embedding.W].remove('trainable')

        embedding_individual = lasagne.layers.ReshapeLayer(embedding,(self.MAX_N_CANDIDATES,self.MAX_LENGTH,self.EMBEDDING_DIM))
        embedding_combined = lasagne.layers.ReshapeLayer(embedding,(1,self.MAX_N_CANDIDATES*self.MAX_LENGTH,self.EMBEDDING_DIM))

        combined_Rep_f = RecurrentLayer(embedding_combined, RECURRENT_N_HID, mask_input=ocm_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        combined_Rep_b = RecurrentLayer(embedding_combined, RECURRENT_N_HID, mask_input=ocm_mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)
        combined_Rep = lasagne.layers.ElemwiseSumLayer([combined_Rep_f,combined_Rep_b])
        combined_Rep_f = RecurrentLayer(combined_Rep, RECURRENT_N_HID, mask_input=ocm_mask, grad_clipping=GRAD_CLIP, only_return_final=True)
        combined_Rep_b = RecurrentLayer(combined_Rep, RECURRENT_N_HID, mask_input=ocm_mask, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        combined_Rep = lasagne.layers.ElemwiseSumLayer([combined_Rep_f,combined_Rep_b])

        print('combined_Rep shape',combined_Rep.output_shape)

        individual_Rep_f = RecurrentLayer(embedding_individual, RECURRENT_N_HID+1, mask_input=mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        individual_Rep_b = RecurrentLayer(embedding_individual, RECURRENT_N_HID+1, mask_input=mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)
        individual_Rep = lasagne.layers.ElemwiseSumLayer([individual_Rep_f,individual_Rep_b])
        individual_Rep_f = RecurrentLayer(individual_Rep, RECURRENT_N_HID+1, mask_input=mask, grad_clipping=GRAD_CLIP, only_return_final=True)
        individual_Rep_b = RecurrentLayer(individual_Rep, RECURRENT_N_HID+1, mask_input=mask, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        individual_Rep = lasagne.layers.ElemwiseSumLayer([individual_Rep_f,individual_Rep_b])

        nc_prediction = lasagne.layers.DenseLayer(combined_Rep, num_units=DENSE_N_HID, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        nc_prediction = lasagne.layers.DropoutLayer(nc_prediction, p=0.5)
        nc_prediction_out = lasagne.layers.DenseLayer(nc_prediction, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

        repeated_combined_Rep = RepeatLayer(combined_Rep,self.MAX_N_CANDIDATES)
        repeated_nc_prediction = RepeatLayer(nc_prediction_out,self.MAX_N_CANDIDATES)

        feature_merge = lasagne.layers.ConcatLayer([individual_Rep,repeated_combined_Rep,repeated_nc_prediction])

        print('feature merge output', feature_merge.output_shape)

        candidate_prediction = lasagne.layers.DenseLayer(feature_merge, num_units=DENSE_N_HID, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        candidate_prediction = lasagne.layers.DropoutLayer(candidate_prediction, p=0.5)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        lm_reshape = lasagne.layers.ReshapeLayer(feature_merge,(self.MAX_N_CANDIDATES*self.MAX_LENGTH, -1))

        print('lm_reshape shape',lm_reshape.output_shape)
        lm = lasagne.layers.DenseLayer(lm_reshape, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
        lm_out = lasagne.layers.ReshapeLayer(lm,(self.MAX_N_CANDIDATES*self.MAX_LENGTH,self.vocab_size))

        print('lm_out shape',lm.output_shape)
        return [candidate_prediction_out,lm_out,nc_prediction_out]
if __name__ == '__main__':
    lstm = Simple_2L('Simple_2L')
    lstm.run()