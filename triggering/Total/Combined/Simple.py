from Base import *

class Simple(Base):
    def __init__(self,name):
        super(Simple, self).__init__(name)

    def define_layers(self,combined_in,masks_in,ocm_in):
        # theano.tensor.extra_ops.cumsum nonzero == num words
        # check how many shared words, dot the bincounts, the get nonzeros / num words

        GRAD_CLIP = 15
        RECURRENT_N_HID = 128
        DENSE_N_HID = 512

        if self.lstm:
            RecurrentLayer = lasagne.layers.LSTMLayer
        else:
            RecurrentLayer = lasagne.layers.GRULayer

        # ----------------------------------------------------- Inputs and Masks ---------------------------------------------------------------------
        combined = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.MAX_LENGTH), input_var=combined_in)
        combined = lasagne.layers.WordDropoutLayer(combined, p=0.5) # will drop EOS tag

        mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.MAX_LENGTH), input_var=masks_in)
        ocm_mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.MAX_LENGTH), input_var=ocm_in)
        ocm_mask = lasagne.layers.ReshapeLayer(ocm_mask, shape=(1,-1))

        # ----------------------------------------------------- Embedding ----------------------------------------------------------------------------

        embedding = lasagne.layers.EmbeddingLayer(combined, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        embedding.params[embedding.W].remove('trainable')

        embedding_individual = lasagne.layers.ReshapeLayer(embedding,(self.MAX_N_CANDIDATES,self.MAX_LENGTH,self.EMBEDDING_DIM))
        embedding_combined = lasagne.layers.ReshapeLayer(embedding,(1,self.MAX_N_CANDIDATES*self.MAX_LENGTH,self.EMBEDDING_DIM))

        # ---------------------------------------------------------RNN-------------------------------------------------------------

        individual_Rep_f = RecurrentLayer(embedding_individual, RECURRENT_N_HID, mask_input=mask, grad_clipping=GRAD_CLIP, only_return_final=True)
        individual_Rep_b = RecurrentLayer(embedding_individual, RECURRENT_N_HID, mask_input=mask, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        individual_Rep = lasagne.layers.ElemwiseSumLayer([individual_Rep_f,individual_Rep_b])

        combined_Rep_f = RecurrentLayer(embedding_combined, 2*RECURRENT_N_HID, mask_input=ocm_mask, grad_clipping=GRAD_CLIP, only_return_final=True)
        combined_Rep_b = RecurrentLayer(embedding_combined, 2*RECURRENT_N_HID, mask_input=ocm_mask, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        combined_Rep = lasagne.layers.ElemwiseSumLayer([combined_Rep_f,combined_Rep_b])

        # -------------------------------------------------- Feature Merging ------------------------------------------------------

        feature_merge = individual_Rep

        # repeated_combined_Rep = RepeatLayer(combined_Rep,self.MAX_N_CANDIDATES)
        # repeated_nc_prediction = RepeatLayer(nc_prediction_out,self.MAX_N_CANDIDATES)
        # feature_merge = lasagne.layers.ConcatLayer([individual_Rep,repeated_combined_Rep])

        candidate_prediction = lasagne.layers.DenseLayer(feature_merge, num_units=DENSE_N_HID, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        candidate_prediction = lasagne.layers.DropoutLayer(candidate_prediction, p=0.5)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        lm_reshape = lasagne.layers.ReshapeLayer(feature_merge,(self.MAX_N_CANDIDATES*self.MAX_LENGTH, -1))

        print('lm_reshape shape',lm_reshape.output_shape)
        lm = lasagne.layers.DenseLayer(lm_reshape, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
        lm_out = lasagne.layers.ReshapeLayer(lm,(self.MAX_N_CANDIDATES*self.MAX_LENGTH,self.vocab_size))

        print('lm_out shape',lm.output_shape)
        return [candidate_prediction_out,lm_out]

if __name__ == '__main__':
    lstm = Simple('Simple')
    lstm.run()