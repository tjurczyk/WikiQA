#CUDA_LAUNCH_BLOCKING=0 THEANO_FLAGS=on_unused_input='ignore',device=gpu1,profile=False python Attention.py  --pbar --pretrain 0 --train 10 --recurrent_layers 1 --candidate_dropout 0.2 --question_dropout 0.2 --lr 1e-3 --l1 1e-6 --recurrent_units 512 --dense_units 512 --trainable_embedding --slice_final
from Base_Sep import Base_Sep
import lasagne
from Layer import RepeatLayer, MergeExpressionLayer, WordDropoutLayer
from lasagne.layers import reshape
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data, slice_out
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, EmbeddingLayer, SliceLayer, ElemwiseSumLayer, Pool1DLayer, ConcatLayer, ReshapeLayer, LSTMLayer, GRULayer, ElemwiseMergeLayer, FeaturePoolLayer
from lasagne.nonlinearities import LeakyRectify
import theano.tensor as T
class Attention(Base_Sep):
    def __init__(self,name):
        super(Attention, self).__init__(name)

    def define_layers(self,questions_in,candidates_in,q_masks_in,c_masks_in):
        GRAD_CLIP = 15
        regularize = {}
        if self.lstm:
            RecurrentLayer = LSTMLayer
        else:
            RecurrentLayer = GRULayer

        question = InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=questions_in)
        if self.question_dropout > 0:
            question = WordDropoutLayer(question, p=self.question_dropout)

        candidates = InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=candidates_in)
        if self.candidate_dropout > 0:
            candidates = WordDropoutLayer(candidates, p=self.candidate_dropout)
        
        question_mask = InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=q_masks_in)
        candidate_mask = InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=c_masks_in)

        # ----------------------------------------------------- Embedding --------------------------------------------------------------

        q_embedding = EmbeddingLayer(question, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        c_embedding = EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        regularize[q_embedding] = 1.
        regularize[c_embedding] = 1.

        if not self.trainable_embedding:
            q_embedding.params[q_embedding.W].remove('trainable')
            c_embedding.params[c_embedding.W].remove('trainable')

        q_embedding = DropoutLayer(q_embedding, p=self.q_embedding_dropout)
        c_embedding = DropoutLayer(c_embedding, p=self.c_embedding_dropout)
        
        # ---------------------------------------------------- question RNN -------------------------------------------------------------
        for _ in xrange(self.q_recurrent_layers):
            q_rep_f = RecurrentLayer(q_embedding, self.recurrent_units, mask_input=question_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        q_feature = q_rep_f #(1, 24, recurrent_units)
        repeat_q_feature = RepeatLayer(q_feature, self.MAX_N_CANDIDATES) # 30, q_maxlen, q_recurrent_units
        q_forward_slice = SliceLayer(q_rep_f, indices=-1, axis=1) #(1, recurrent_units)
        repeated_question = RepeatLayer(q_forward_slice, self.MAX_N_CANDIDATES)
        # ---------------------------------------------------- candidate RNN -------------------------------------------------------------
        c_rep_f = RecurrentLayer(c_embedding, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        if self.bidirectional:
            c_rep_b = RecurrentLayer(c_embedding, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)

            c_feature = ElemwiseSumLayer([c_rep_f,c_rep_b]) # (30,40,c_recurrent Units)
            c_rep_f = c_feature
            c_rep_b = c_feature
        else: # in case we use only 1 layer
            c_feature = c_rep_f # (30,40,c_recurrent Units)

        if self.c_recurrent_layers > 1:
            for _ in xrange(1,self.c_recurrent_layers):
                c_rep_f = RecurrentLayer(c_rep_f, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
                if self.bidirectional:
                    c_rep_b = RecurrentLayer(c_rep_b, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)

                    c_feature = ElemwiseSumLayer([c_rep_f,c_rep_b]) # (30,40,c_recurrent Units)
                    c_rep_f = c_feature
                    c_rep_b = c_feature

        if self.bidir_concat:
            c_rep_length = 2*self.C_MAX_LENGTH
            c_feature = ConcatLayer([c_rep_f,c_rep_b]) # (30,80,c_recurrent Units)
        else:
            c_rep_length = self.C_MAX_LENGTH

        # ------------------------------------------------------ Final Attention -------------------------------------------------------------
        if self.embedding_attention:
            q_attention_embedding = EmbeddingLayer(question, input_size=self.vocab_size, output_size=10)
            regularize[q_attention_embedding] = 1.
            repeat_q_feature = RepeatLayer(q_attention_embedding, repeats=self.MAX_N_CANDIDATES)
            c_feature = EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=10)
            regularize[c_feature] = 1.
        if self.attention_units > 0:
            combined = ConcatLayer([repeat_q_feature,c_feature])
            match = DenseLayer(combined, num_units=self.attention_units, nonlinearity=lasagne.nonlinearities.tanh)
            if self.attention_dropout > 0:
                match = DropoutLayer(match, p=self.attention_dropout)
        else:
            match = MergeExpressionLayer([repeat_q_feature,c_feature], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(self.MAX_N_CANDIDATES,self.Q_MAX_LENGTH,c_rep_length))
        attention = DenseLayer(match, num_units=c_rep_length, nonlinearity=lasagne.nonlinearities.softmax)
        attention = RepeatLayer(ReshapeLayer(attention, (-1, c_rep_length, 1)),repeats=self.recurrent_units, axis=2)
        attended = ElemwiseMergeLayer([attention, c_feature], T.mul)

        # -------------------------------------------------- Candidate Prediction -------------------------------------------------
        reshaped_question = ReshapeLayer(RepeatLayer(q_forward_slice,repeats=self.MAX_N_CANDIDATES), (self.MAX_N_CANDIDATES, 1, self.recurrent_units))
        feature = ConcatLayer([reshaped_question, attended], axis=1)

        if self.out_recurrent_units > 0:
            feature = RecurrentLayer(feature, num_units=self.out_recurrent_units, only_return_final=True)
            if self.out_recurrent_dropout > 0:
                feature = DropoutLayer(feature, p=self.out_recurrent_dropout)
        if self.dense_units > 0:
            feature = DenseLayer(feature, num_units=self.dense_units, nonlinearity=lasagne.nonlinearities.tanh)
            if self.dense_dropout > 0:
                feature = DropoutLayer(feature, p=self.dense_dropout)
        candidate_prediction_out = DenseLayer(feature, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        
        # ---------------------------------------------------------- Level 2 ------------------------------------------------------
        
        q_feature = DenseLayer(feature, num_units=self.recurrent_units, nonlinearity=lasagne.nonlinearities.sigmoid)
        repeat_q_feature = ReshapeLayer(RepeatLayer(FeaturePoolLayer(q_feature, pool_size=self.MAX_N_CANDIDATES, axis=0, pool_function=T.max),self.MAX_N_CANDIDATES),(self.MAX_N_CANDIDATES,-1,self.recurrent_units))
        feature = ConcatLayer([repeat_q_feature, attended], axis=1)
        if self.out_recurrent_units > 0:
            feature = RecurrentLayer(feature, num_units=self.out_recurrent_units, only_return_final=True)
            if self.out_recurrent_dropout > 0:
                feature = DropoutLayer(feature, p=self.out_recurrent_dropout)
        if self.dense_units > 0:
            feature = DenseLayer(feature, num_units=self.dense_units, nonlinearity=lasagne.nonlinearities.tanh)
            if self.dense_dropout > 0:
                feature = DropoutLayer(feature, p=self.dense_dropout)

        print('feature',feature.output_shape)
        candidate_prediction_out_2 = DenseLayer(feature, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        # ------------------------------------------------------ LM Pretraining ---------------------------------------------------
        
        if self.end_p_epoch > 0:
            lm_target = ConcatLayer([repeated_question,candidates])
            lm_reshape = ReshapeLayer(feature,(self.MAX_N_CANDIDATES,1, -1))
            lm_decoder = RecurrentLayer(lm_reshape, 100, grad_clipping=GRAD_CLIP, only_return_final=False)
            lm_out = DenseLayer(lm_decoder, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
            return [candidate_prediction_out, lm_out, lm_target, regularize]

        return [candidate_prediction_out_2, regularize]
if __name__ == '__main__':
    lstm = Attention('Attention')
    lstm.get_args()
    lstm.run()