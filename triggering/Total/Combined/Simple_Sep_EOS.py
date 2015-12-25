from Base_Sep import *
# CUDA_LAUNCH_BLOCKING=0 THEANO_FLAGS=on_unused_input='ignore',device=gpu0,optimizer=fast_run,exception_verbosity='low',allow_gc=False,profile=False python Simple_Sep_EOS.py -p 5 -t 100 -e -s -l 2 --layers 3 --rename 3layer

class Simple_Sep_EOS(Base_Sep):
    def __init__(self,name):
        super(Simple_Sep_EOS, self).__init__(name)

    def define_layers(self,questions_in,candidates_in,q_masks_in,c_masks_in):
        GRAD_CLIP = 15
        if self.lstm:
            RecurrentLayer = lasagne.layers.LSTMLayer
        else:
            RecurrentLayer = lasagne.layers.GRULayer

        question = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=questions_in)
        question = lasagne.layers.WordDropoutLayer(question, p=self.question_dropout)
        question_EOS = lasagne.layers.PadLayer(question, width=[(0,0),(0,1)],val=1,batch_ndim=0,dtype='int32')
        repeated_question = RepeatLayer(question_EOS,self.MAX_N_CANDIDATES)
        candidates = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=candidates_in)
        candidates = lasagne.layers.WordDropoutLayer(candidates, p=self.candidate_dropout)
        combined = lasagne.layers.ConcatLayer([repeated_question, candidates])

        question_mask = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=q_masks_in)
        question_mask_EOS = lasagne.layers.PadLayer(question_mask, width=[(0,0),(0,1)],val=1,batch_ndim=0,dtype='float32')
        repeated_question_mask = RepeatLayer(question_mask_EOS,self.MAX_N_CANDIDATES)
        candidate_mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=c_masks_in)
        combined_mask = lasagne.layers.ConcatLayer([repeated_question_mask,candidate_mask])

        # theano.tensor.extra_ops.cumsum nonzero == num words
        # check how many shared words, dot the bincounts, the get nonzeros / num words
        # ----------------------------------------------------- Embedding --------------------------------------------------------
        embedding = lasagne.layers.EmbeddingLayer(combined, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        if not self.trainable_embedding:
            embedding.params[embedding.W].remove('trainable')

        # ---------------------------------------------------------RNN-------------------------------------------------------------
        
        for _ in xrange(self.n_layers):
            Rep_f = RecurrentLayer(embedding, N_HIDDEN, mask_input=combined_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
            Rep_b = RecurrentLayer(embedding, N_HIDDEN, mask_input=combined_mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)
            embedding = lasagne.layers.ElemwiseSumLayer([Rep_f,Rep_b])
        final_rep = embedding
        features = []
        if self.slice_final:
            features.append(lasagne.layers.SliceLayer(final_rep, indices=-1, axis=1))
        if self.mean_pool:
            features.append(lasagne.layers.Pool1DLayer(final_rep,pool_size=embedding.output_shape[1],ignore_border=False,mode='average_inc_pad'))
        if self.max_pool:
            features.append(lasagne.layers.Pool1DLayer(final_rep,pool_size=embedding.output_shape[1],ignore_border=False,mode='max'))
        
        # -------------------------------------------------- Feature Merging ------------------------------------------------------
        for f in features:
            print(f.output_shape)
        if len(features) > 1:
            feature_merge = lasagne.layers.ConcatLayer(features)
        else:
            feature_merge = features[0]
        
        # -------------------------------------------------- Candidate Prediction -------------------------------------------------

        candidate_prediction = lasagne.layers.DenseLayer(feature_merge, num_units=self.D_HIDDEN, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        candidate_prediction = lasagne.layers.DropoutLayer(candidate_prediction, p=self.hid_dropout)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        
        # ------------------------------------------------------ LM Pretraining ---------------------------------------------------
        
        lm_reshape = lasagne.layers.ReshapeLayer(final_rep,(self.MAX_N_CANDIDATES*(self.Q_MAX_LENGTH+self.C_MAX_LENGTH), -1))
        lm = lasagne.layers.DenseLayer(lm_reshape, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
        lm_out = lasagne.layers.ReshapeLayer(lm,(self.MAX_N_CANDIDATES*(self.Q_MAX_LENGTH+self.C_MAX_LENGTH),self.vocab_size))
        return [candidate_prediction_out, lm_out]

if __name__ == '__main__':
    lstm = Simple_Sep_EOS('Simple_Sep_EOS')
    lstm.run()