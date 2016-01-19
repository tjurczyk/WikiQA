#CUDA_LAUNCH_BLOCKING=0 THEANO_FLAGS=on_unused_input='ignore',device=gpu1,profile=False python Attention.py  --pbar --pretrain 0 --train 10 --recurrent_layers 1 --candidate_dropout 0.2 --question_dropout 0.2 --lr 1e-3 --l1 1e-6 --recurrent_units 512 --dense_units 512 --trainable_embedding --slice_final
from Base_Sep import Base_Sep
import lasagne
from Layer import RepeatLayer, MergeExpressionLayer, WordDropoutLayer
from lasagne.layers import reshape
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data, slice_out
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, EmbeddingLayer, SliceLayer, ElemwiseSumLayer, Pool1DLayer, ConcatLayer, ReshapeLayer, ElemwiseMergeLayer, FeaturePoolLayer, LSTMLayer, GRULayer
# from lasagne.nonlinearities import LeakyRectify
# from recurrent import LSTMLayer, GRULayer
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
        if self.question_dropout > -1:
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
        

        # --------------------------------------------------Embedding Attention ---------------------------------------------------------
        # q_rep = DropoutLayer(RecurrentLayer(q_embedding, self.recurrent_units, mask_input=question_mask, grad_clipping=GRAD_CLIP, only_return_final=False),p=self.q_recurrent_dropout)
        # q_rep = RepeatLayer(q_rep,repeats=self.MAX_N_CANDIDATES,axis=0)
        # c_rep = DropoutLayer(RecurrentLayer(c_embedding, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False),p=self.c_recurrent_dropout)
        # match = MergeExpressionLayer([q_rep,c_rep], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(self.MAX_N_CANDIDATES,self.Q_MAX_LENGTH,self.C_MAX_LENGTH))
        # attention = DenseLayer(match, num_units=self.C_MAX_LENGTH, nonlinearity=lasagne.nonlinearities.softmax)
        # print(attention.output_shape)
        # attention = RepeatLayer(ReshapeLayer(attention,(self.MAX_N_CANDIDATES,-1,1)),repeats=self.EMBEDDING_DIM,axis=2)
        # print(attention.output_shape)
        # c_embedding = ElemwiseMergeLayer([attention, c_embedding], T.mul)
        # print(c_embedding.output_shape)

        # ---------------------------------------------------- question RNN -------------------------------------------------------------
        for _ in xrange(self.q_recurrent_layers):
            q_rep_f = RecurrentLayer(q_embedding, self.recurrent_units, mask_input=question_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
            if self.q_recurrent_dropout > 0:
                q_rep_f = DropoutLayer(q_rep_f,p=self.q_recurrent_dropout)
        q_feature = q_rep_f #(1, 24, recurrent_units)
        repeat_q_feature = RepeatLayer(q_feature, self.MAX_N_CANDIDATES) # 30, q_maxlen, q_recurrent_units
        q_forward_slice = SliceLayer(q_rep_f, indices=-1, axis=1) #(1, recurrent_units)
        repeated_q_slice = RepeatLayer(q_forward_slice, self.MAX_N_CANDIDATES)

        # ------------------------------------------------------ candidate encoder -------------------------------------------------------------

        c_rep_f = RecurrentLayer(c_embedding, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False)
        if self.c_recurrent_dropout > 0:
            c_rep_f = DropoutLayer(c_rep_f,p=self.c_recurrent_dropout)
        if self.bidirectional:
            c_rep_b = RecurrentLayer(c_embedding, self.recurrent_units, mask_input=candidate_mask, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)
            if self.c_recurrent_dropout > 0:
                c_rep_b = DropoutLayer(c_rep_b,p=self.c_recurrent_dropout)
            c_feature = ElemwiseSumLayer([c_rep_f,c_rep_b]) # (30,40,c_recurrent Units)
        else:
            c_feature = c_rep_f
        c_rep_length = self.C_MAX_LENGTH

        # --------------------------------------------------- candidate encoder attention ------------------------------------------------------------------

        if self.encoder_attention:
            match = MergeExpressionLayer([repeat_q_feature,c_feature], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(self.MAX_N_CANDIDATES,self.Q_MAX_LENGTH,self.C_MAX_LENGTH))
            attention = DenseLayer(match, num_units=self.C_MAX_LENGTH, nonlinearity=lasagne.nonlinearities.softmax)
            # attention = RepeatLayer(ReshapeLayer(attention,(self.MAX_N_CANDIDATES,-1,1)),repeats=self.recurrent_units,axis=2)
            # c_rep_f = ElemwiseMergeLayer([attention, c_rep_f], T.mul)
            # if self.bidirectional:
            #     c_rep_b = ElemwiseMergeLayer([attention, c_rep_b], T.mul)
        else:
            attention = None

        # -------------------------------------------------------- candidate encoder ------------------------------------------------------------------

        for _ in xrange(1,self.c_recurrent_layers):
            c_rep_f = RecurrentLayer(c_rep_f, self.recurrent_units, mask_input=candidate_mask, attention=attention, grad_clipping=GRAD_CLIP, only_return_final=False)
            if self.c_recurrent_dropout > 0:
                c_rep_f = DropoutLayer(c_rep_f,p=self.c_recurrent_dropout)
            if self.bidirectional:
                c_rep_b = RecurrentLayer(c_rep_b, self.recurrent_units, mask_input=candidate_mask, attention=attention, grad_clipping=GRAD_CLIP, only_return_final=False, backwards=True)
                if self.c_recurrent_dropout > 0:
                    c_rep_b = DropoutLayer(c_rep_b,p=self.c_recurrent_dropout)
                if self.sum_intermediate:
                    c_feature = ElemwiseSumLayer([c_rep_f,c_rep_b]) # (30,40,c_recurrent Units)
                    c_rep_f = c_feature
                    c_rep_b = c_feature
            else: # in case we use only 1 layer
                c_feature = c_rep_f # (30,40,c_recurrent Units)

        # ------------------------------------------------------- decoder attention -------------------------------------------------------------

        if self.bidir_concat:
            c_rep_length = 2*self.C_MAX_LENGTH
            c_feature = ConcatLayer([c_rep_b,c_rep_f]) # if slice take 0, -1
        else:
            c_rep_length = self.C_MAX_LENGTH
        
        if self.decoder_attention:
            if self.embedding_attention:
                q_attention_rep = RepeatLayer(EmbeddingLayer(question, input_size=self.vocab_size, output_size=self.recurrent_units),repeats=self.MAX_N_CANDIDATES)
                regularize[q_attention_rep] = 1.
                c_attention_rep = EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=self.recurrent_units)
                regularize[c_attention_rep] = 1.
                if self.attention_embedding_dropout > 0:
                    q_attention_rep = DropoutLayer(q_attention_rep, p=self.attention_embedding_dropout)
                    c_attention_rep = DropoutLayer(c_attention_rep, p=self.attention_embedding_dropout)
            else:
                c_attention_rep = c_feature
                q_attention_rep = repeat_q_feature

            if self.attention_units > 0:
                combined = ConcatLayer([q_attention_rep,c_attention_rep])
                match = DenseLayer(combined, num_units=self.attention_units, nonlinearity=lasagne.nonlinearities.tanh)
                if self.attention_dropout > 0:
                    match = DropoutLayer(match, p=self.attention_dropout)
            else:
                match = MergeExpressionLayer([q_attention_rep,c_attention_rep], lambda X,Y: T.batched_tensordot(X,Y,axes=[2,2]), output_shape=(self.MAX_N_CANDIDATES,self.Q_MAX_LENGTH,c_rep_length))
            
            attention = DenseLayer(match, num_units=c_rep_length, nonlinearity=lasagne.nonlinearities.softmax)
            attention = RepeatLayer(ReshapeLayer(attention, (-1, c_rep_length, 1)),repeats=self.recurrent_units, axis=2)
            attended = ElemwiseMergeLayer([attention, c_feature], T.mul)

            if self.attention_question:
                q_attention = DenseLayer(match, num_units=self.Q_MAX_LENGTH, nonlinearity=lasagne.nonlinearities.softmax)
                q_attention = RepeatLayer(ReshapeLayer(q_attention, (-1, self.Q_MAX_LENGTH, 1)),repeats=self.recurrent_units, axis=2)
                q_attention = RepeatLayer(FeaturePoolLayer(q_attention, pool_size=self.MAX_N_CANDIDATES, axis=0, pool_function=T.mean),repeats=self.MAX_N_CANDIDATES, axis=0)
                q_attended = ElemwiseMergeLayer([q_attention, repeat_q_feature], T.mul)
                feature = ConcatLayer([q_attended, attended], axis=1)
            else:
                reshaped_question = ReshapeLayer(repeated_q_slice, (self.MAX_N_CANDIDATES, 1, self.recurrent_units))
                feature = ConcatLayer([reshaped_question, attended], axis=1)
        else:
            reshaped_question = ReshapeLayer(repeated_q_slice, (self.MAX_N_CANDIDATES, 1, self.recurrent_units))
            c_forward_slice = ReshapeLayer(SliceLayer(c_feature, indices=-1, axis=1), (self.MAX_N_CANDIDATES, 1, self.recurrent_units))
            if self.bidirectional:
                c_backward_slice = ReshapeLayer(SliceLayer(c_feature, indices=0, axis=1), (self.MAX_N_CANDIDATES, 1, self.recurrent_units))
                feature = ConcatLayer([reshaped_question, c_forward_slice, c_backward_slice], axis=1)
            else:
                feature = ConcatLayer([reshaped_question, c_forward_slice], axis=1)

        # ------------------------------------------------------- episodic memory ------------------------------------------------------
        # print('feature',feature.output_shape)
        # episodes = ReshapeLayer(attended, (1,30,-1))
        # # episodes = ReshapeLayer(feature, (1,30,-1))
        # print('episodes',episodes.output_shape)
        # episode_gate = RecurrentLayer(episodes, num_units=self.recurrent_units, hid_init=q_feature, only_return_final=True)
        # print('episode_gate',episode_gate.output_shape)
        # episode_weights = DenseLayer(episode_gate, num_units=self.MAX_N_CANDIDATES, nonlinearity=lasagne.nonlinearities.softmax)
        # print('episode_weights',episode_weights.output_shape)
        # episode_out = RecurrentLayer(episodes, num_units=self.recurrent_units*5, attention=episode_weights, only_return_final=False)
        # print('feature',feature.output_shape)
        # candidate_prediction_out = ReshapeLayer(DenseLayer(episode_out, num_units=self.MAX_N_CANDIDATES, nonlinearity=lasagne.nonlinearities.softmax),(30,1))
        # print(candidate_prediction_out.output_shape)
        # -------------------------------------------------- Candidate Prediction -------------------------------------------------
        if self.out_recurrent_units > 0:
            feature = RecurrentLayer(feature, num_units=self.out_recurrent_units, only_return_final=True)
            if self.out_recurrent_dropout > 0:
                feature = DropoutLayer(feature, p=self.out_recurrent_dropout)
        if self.dense_units > 0:
            feature = DenseLayer(feature, num_units=self.dense_units, nonlinearity=lasagne.nonlinearities.tanh)
            if self.dense_dropout > 0:
                feature = DropoutLayer(feature, p=self.dense_dropout)
        if self.hinge:
            out_activation = lasagne.nonlinearities.linear
        else:
            out_activation = lasagne.nonlinearities.sigmoid
        candidate_prediction_out = DenseLayer(feature, num_units=1, nonlinearity=out_activation)
        
        # ---------------------------------------------------------- Level 2 ------------------------------------------------------
        
        # if self.deep:
        if False:
            q_feature = DenseLayer(feature, num_units=self.recurrent_units, nonlinearity=lasagne.nonlinearities.tanh)
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