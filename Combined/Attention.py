#CUDA_LAUNCH_BLOCKING=0 THEANO_FLAGS=on_unused_input='ignore',device=gpu1,profile=False python Attention.py  --pbar --pretrain 0 --train 10 --recurrent_layers 1 --candidate_dropout 0.2 --question_dropout 0.2 --lr 1e-3 --l1 1e-6 --recurrent_units 512 --dense_units 512 --trainable_embedding --slice_final
from Base_Sep import Base_Sep
import lasagne
from Layer import RepeatLayer, MergeExpressionLayer
from lasagne.layers import reshape
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data, slice_out
from lasagne.layers import DenseLayer, InputLayer, WordDropoutLayer, DropoutLayer, EmbeddingLayer, SliceLayer, ElemwiseSumLayer, Pool1DLayer, ConcatLayer, ReshapeLayer, LSTMLayer, GRULayer, ElemwiseMergeLayer
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
        for _ in xrange(self.c_recurrent_layers):
            c_rep_f = RecurrentLayer(c_embedding, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        c_feature= c_rep_f # (30,40,c_recurrent Units)

        # ----------------------------------------------- Attention candidate RNN -------------------------------------------------------------
        # self.c_recurrent_units = 50
        # for i in xrange(1,self.recurrent_layers+1):
        #     c_rep_f = RecurrentLayer(c_embedding, self.c_recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        #     dot = MergeExpressionLayer([repeat_q_feature,c_rep_f], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(30,self.Q_MAX_LENGTH,self.C_MAX_LENGTH))
        #     if self.attention_units > 0:
        #         if self.attention_dropout > 0:
        #             dot = DropoutLayer(DenseLayer(dot, num_units=self.attention_units, nonlinearity=lasagne.nonlinearities.tanh), p=self.attention_dropout)
        #         else:
        #             dot = DenseLayer(dot, num_units=self.attention_units, nonlinearity=lasagne.nonlinearities.tanh)
        #     attention = DenseLayer(dot, num_units=self.C_MAX_LENGTH, nonlinearity=lasagne.nonlinearities.softmax)
        #     attention = RepeatLayer(ReshapeLayer(attention, (-1, self.C_MAX_LENGTH, 1)),repeats=self.c_recurrent_units, axis=2)
        #     c_rep_f = ElemwiseMergeLayer([attention, c_rep_f], T.mul)
        # attended_c_feature = c_rep_f # (30,40,c_recurrent Units)

        # ------------------------------------------------------ Final Attention -------------------------------------------------------------
        if self.embedding_attention:
            q_attention_embedding = EmbeddingLayer(WordDropoutLayer(question,p=0.5), input_size=self.vocab_size, output_size=10)
            regularize[q_attention_embedding] = 1.
            q_attention_embedding = RepeatLayer(q_attention_embedding, repeats=self.MAX_N_CANDIDATES)
            c_attention_embedding = EmbeddingLayer(WordDropoutLayer(candidates, p=0.5), input_size=self.vocab_size, output_size=10)
            regularize[c_attention_embedding] = 1.
            dot = MergeExpressionLayer([q_attention_embedding,c_attention_embedding], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(self.MAX_N_CANDIDATES,self.Q_MAX_LENGTH,self.C_MAX_LENGTH))
        else:
            dot = MergeExpressionLayer([repeat_q_feature,c_feature], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(self.MAX_N_CANDIDATES,self.Q_MAX_LENGTH,self.C_MAX_LENGTH))
        if self.attention_units > 0:
            dot = DenseLayer(dot, num_units=self.attention_units, nonlinearity=lasagne.nonlinearities.tanh)
            if self.attention_dropout > 0:
                dot = DropoutLayer(dot, p=self.attention_dropout)

        attention = DenseLayer(dot, num_units=self.C_MAX_LENGTH, nonlinearity=lasagne.nonlinearities.softmax)
        print('attention',attention.output_shape)
        attention = RepeatLayer(ReshapeLayer(attention, (-1, self.C_MAX_LENGTH, 1)),repeats=self.recurrent_units, axis=2)
        c_feature = ElemwiseMergeLayer([attention, c_feature], T.mul)

        # attention = DenseLayer(dot, num_units=self.recurrent_units, nonlinearity=lasagne.nonlinearities.softmax)
        # attention = RepeatLayer(ReshapeLayer(attention, (-1, 1, self.recurrent_units)),repeats=self.C_MAX_LENGTH, axis=1)
        # c_feature = ElemwiseMergeLayer([attention, c_feature], T.mul)
        # print('attention',attention.output_shape)

        # ----------------------------------------------- Global Candidate Prediction -------------------------------------------------------------

        # for i in xrange(self.MAX_N_CANDIDATES):
        #     out, rest = slice_out(c_feature, index=i, axis=0)
        #     print('out',out.output_shape) #(1, 40, 50)
        #     print('rest',rest.output_shape) #(29, 40, 50)
        # -------------------------------------------------- Candidate Prediction -------------------------------------------------
        reshaped_question = ReshapeLayer(RepeatLayer(q_forward_slice,repeats=self.MAX_N_CANDIDATES), (self.MAX_N_CANDIDATES, 1, self.recurrent_units))
        print('reshaped_question', reshaped_question.output_shape)
        feature = ConcatLayer([reshaped_question, c_feature], axis=1)
        print('feature', feature.output_shape)

        if self.out_recurrent_units > 0:
            feature = RecurrentLayer(feature, num_units=self.out_recurrent_units, only_return_final=True)
            if self.out_recurrent_dropout > 0:
                feature = DropoutLayer(feature, p=self.out_recurrent_dropout)
        if self.dense_units > 0:
            feature = DropoutLayer(DenseLayer(feature, num_units=self.dense_units, nonlinearity=lasagne.nonlinearities.tanh), p=self.dense_dropout)
        candidate_prediction_out = DenseLayer(feature, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        
        # ------------------------------------------------------ LM Pretraining ---------------------------------------------------
        
        if self.end_p_epoch > 0:
            lm_target = ConcatLayer([repeated_question,candidates])
            lm_reshape = ReshapeLayer(feature,(self.MAX_N_CANDIDATES,1, -1))
            lm_decoder = RecurrentLayer(lm_reshape, 100, grad_clipping=GRAD_CLIP, only_return_final=False)
            lm_out = DenseLayer(lm_decoder, num_units=self.vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
            return [candidate_prediction_out, lm_out, lm_target, regularize]

        return [candidate_prediction_out, regularize]
if __name__ == '__main__':
    lstm = Attention('Attention')
    lstm.get_args()
    lstm.run()