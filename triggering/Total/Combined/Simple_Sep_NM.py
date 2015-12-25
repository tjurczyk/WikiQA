from Base_Sep import *

class Simple_Sep_NM(Base_Sep):
    def __init__(self,name):
        super(Simple_Sep_NM, self).__init__(name)

    def define_layers(self,questions_in,candidates_in,q_masks_in,c_masks_in):
        Q_N_HIDDEN = 128

        GRAD_CLIP = 10
        GRAD_STEPS = 100

        C_N_HIDDEN = 128
        question = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=questions_in)
        candidates = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=candidates_in)

        # theano.tensor.extra_ops.cumsum nonzero == num words
        # check how many shared words, dot the bincounts, the get nonzeros / num words

        # ---------------------------------------------------------Embeddings ----------------------------------------------------
        q_embedding = lasagne.layers.EmbeddingLayer(question, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        c_embedding = lasagne.layers.EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=q_embedding.W)

        q_embedding.params[q_embedding.W].remove('trainable')
        c_embedding.params[c_embedding.W].remove('trainable')

        # regular embedding
        q_embedding_reshape = lasagne.layers.ReshapeLayer(q_embedding,(1,self.Q_MAX_LENGTH,self.EMBEDDING_DIM))
        c_embedding_individual = lasagne.layers.ReshapeLayer(c_embedding,(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH,self.EMBEDDING_DIM))
        c_embedding_combined = lasagne.layers.ReshapeLayer(c_embedding,(1,self.MAX_N_CANDIDATES*self.C_MAX_LENGTH,self.EMBEDDING_DIM))

        # ---------------------------------------------------------RNN-------------------------------------------------------------
        qRep = lasagne.layers.GRULayer(q_embedding_reshape, Q_N_HIDDEN, grad_clipping=GRAD_CLIP, only_return_final=True)

        c_combined_Rep_f = lasagne.layers.GRULayer(c_embedding_combined, 2*C_N_HIDDEN, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_combined_Rep_b = lasagne.layers.GRULayer(c_embedding_combined, 2*C_N_HIDDEN, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_combined_Rep = lasagne.layers.ElemwiseSumLayer([c_combined_Rep_f,c_combined_Rep_b])

        c_individual_Rep_f = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_individual_Rep_b = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_individual_Rep = lasagne.layers.ElemwiseSumLayer([c_individual_Rep_f,c_individual_Rep_b])

        # -------------------------------------------------- Feature Merging ------------------------------------------------------

        # repeated_qRep = RepeatLayer(qRep,self.MAX_N_CANDIDATES)
        # repeated_c_combined_Rep = RepeatLayer(c_combined_Rep,self.MAX_N_CANDIDATES)
        # feature_merge = lasagne.layers.ConcatLayer([repeated_qRep,c_individual_Rep,repeated_c_combined_Rep])
        feature_merge = c_individual_Rep
        # -------------------------------------------------- Candidate Prediction -------------------------------------------------
        candidate_prediction = lasagne.layers.DenseLayer(feature_merge, num_units=512, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        candidate_prediction = lasagne.layers.DropoutLayer(candidate_prediction, p=0.5)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        
        # ------------------------------------------------------ LM Pretraining ---------------------------------------------------

        lm_reshape = lasagne.layers.ReshapeLayer(feature_merge,(self.MAX_N_CANDIDATES*(self.Q_MAX_LENGTH+self.C_MAX_LENGTH), -1))
        print('lm_reshape shape',lm_reshape.output_shape)
        lm = lasagne.layers.DenseLayer(lm_reshape, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
        lm_out = lasagne.layers.ReshapeLayer(lm,(self.MAX_N_CANDIDATES*(self.Q_MAX_LENGTH+self.C_MAX_LENGTH),self.vocab_size))

        print('candidate_prediction_out output', candidate_prediction_out.output_shape)
        return [candidate_prediction_out, lm_out]
if __name__ == '__main__':
    lstm = Simple_Sep_NM('Simple_Sep_NM')
    lstm.run()