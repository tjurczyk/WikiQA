#CUDA_LAUNCH_BLOCKING=0 THEANO_FLAGS=on_unused_input='ignore',device=gpu1,profile=False python Simple_Sep_EOS.py  --pbar --pretrain 0 --train 10 --recurrent_layers 1 --candidate_dropout 0.2 --question_dropout 0.2 --lr 1e-3 --l1 1e-6 --recurrent_units 512 --dense_units 512 --trainable_embedding --slice_final
from Base_Sep import Base_Sep
import lasagne
from Layer import RepeatLayer, WordDropoutLayer
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data
from losses import define_lm_losses, define_candidate_losses

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
        if self.question_dropout > 0:
            question = WordDropoutLayer(question, p=self.question_dropout)
        question_EOS = lasagne.layers.PadLayer(question, width=[(0,0),(0,1)],val=1,batch_ndim=0,dtype='int32')
        repeated_question = RepeatLayer(question_EOS,self.MAX_N_CANDIDATES)
        candidates = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=candidates_in)
        if self.candidate_dropout > 0:
            candidates = WordDropoutLayer(candidates, p=self.candidate_dropout)
        question_mask = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=q_masks_in)
        question_mask_EOS = lasagne.layers.PadLayer(question_mask, width=[(0,0),(0,1)],val=1,batch_ndim=0,dtype='float32')
        repeated_question_mask = RepeatLayer(question_mask_EOS,self.MAX_N_CANDIDATES)
        candidate_mask = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=c_masks_in)
        combined_mask = lasagne.layers.ConcatLayer([repeated_question_mask,candidate_mask])

        # theano.tensor.extra_ops.cumsum nonzero == num words
        # check how many shared words, dot the bincounts, the get nonzeros / num words
        # ----------------------------------------------------- Embedding --------------------------------------------------------

        embedding_q = lasagne.layers.EmbeddingLayer(repeated_question, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        embedding_c = lasagne.layers.EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)

        if not self.trainable_embedding:
            embedding_q.params[embedding_q.W].remove('trainable')
            embedding_c.params[embedding_c.W].remove('trainable')
        
        embedding_q = lasagne.layers.DropoutLayer(embedding_q, p=self.q_embedding_dropout)
        embedding_c = lasagne.layers.DropoutLayer(embedding_c, p=self.c_embedding_dropout)

        combined = lasagne.layers.ConcatLayer([embedding_q, embedding_c])

        # ---------------------------------------------------------RNN-------------------------------------------------------------
        rep_f = None
        for _ in xrange(self.recurrent_layers):
            rep_f = RecurrentLayer(combined, self.recurrent_units, mask_input=combined_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
            rep_b = None
            if self.bidirectional:
                rep_b = RecurrentLayer(combined, self.recurrent_units, mask_input=combined_mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)
                combined = lasagne.layers.ElemwiseSumLayer([rep_f,rep_b])
            else:
                combined = rep_f
        feature = None
        if self.slice_final:
            forward_slice = lasagne.layers.SliceLayer(rep_f, indices=-1, axis=1)
            feature = forward_slice
            if self.bidirectional:
                backward_slice = lasagne.layers.SliceLayer(rep_b, indices=0, axis=1)
                feature = lasagne.layers.ConcatLayer([forward_slice,backward_slice])
        elif self.average_pool:
            feature = lasagne.layers.Pool1DLayer(combined, pool_size=self.average_pool_size, ignore_border=False, mode='average_exc_pad')
        elif self.max_pool:
            feature = lasagne.layers.Pool1DLayer(combined, pool_size=self.max_pool_size, ignore_border=False, mode='max')
        
        # -------------------------------------------------- Candidate Prediction -------------------------------------------------

        if self.dense_units > 0:
            feature = lasagne.layers.DropoutLayer(feature, p=self.dense_dropout)
            leaky = lasagne.nonlinearities.LeakyRectify(leakiness=self.leakiness)
            feature = lasagne.layers.DenseLayer(feature, num_units=self.dense_units, nonlinearity=leaky)
        candidate_prediction_out = lasagne.layers.DenseLayer(feature, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print('candidate_prediction_out target',candidate_prediction_out.output_shape)

        # ------------------------------------------------------ LM Pretraining ---------------------------------------------------
        if self.end_p_epoch > 0:
            lm_target = lasagne.layers.ConcatLayer([repeated_question,candidates])
            lm_reshape = lasagne.layers.ReshapeLayer(feature,(self.MAX_N_CANDIDATES*(self.Q_MAX_LENGTH+1+self.C_MAX_LENGTH), -1, 1))
            # lm = lasagne.layers.DenseLayer(lm_reshape, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
            # lm_out = lasagne.layers.ReshapeLayer(lm,(self.MAX_N_CANDIDATES*(self.Q_MAX_LENGTH+self.C_MAX_LENGTH),self.vocab_size))
            lm_decoder = RecurrentLayer(lm_reshape, 100, grad_clipping=GRAD_CLIP, only_return_final=True)
            lm_out = lasagne.layers.DenseLayer(lm_decoder, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
            return [candidate_prediction_out, lm_out, lm_target]

        return [candidate_prediction_out]

if __name__ == '__main__':
    lstm = Simple_Sep_EOS('Simple_Sep_EOS')
    lstm.get_args()
    lstm.run()