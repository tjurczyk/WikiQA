# from base import *
from DynamicBias import *
from lasagne.regularization import l2, l1
import lasagne

class Single(DynamicBias):
    def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
        super(Single, self).__init__(name,macro_batch_size,micro_batch_size,end_epoch)

    def define_layers(self,questions_in,candidates_in,q_masks_in,c_masks_in):
        Q_N_HIDDEN = 100

        GRAD_CLIP = 10
        GRAD_STEPS = 40

        C_N_HIDDEN = 100
        SUBSPACE_DIM = 100
        question = lasagne.layers.InputLayer(shape=(1,self.Q_MAX_LENGTH), input_var=questions_in)
        candidates = lasagne.layers.InputLayer(shape=(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), input_var=candidates_in)

        q_embedding = lasagne.layers.EmbeddingLayer(question, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=self.embedding)
        c_embedding = lasagne.layers.EmbeddingLayer(candidates, input_size=self.vocab_size, output_size=self.EMBEDDING_DIM, W=q_embedding.W)

        q_embedding.params[q_embedding.W].remove('trainable')
        c_embedding.params[c_embedding.W].remove('trainable')

        q_single = lasagne.layers.DenseLayer(q_embedding, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        repeated_q_single = RepeatLayer(q_single,self.MAX_N_CANDIDATES)

        c_single = lasagne.layers.DenseLayer(c_embedding, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print('q_single',q_single.output_shape)
        print('c_single',c_single.output_shape)
        c_single_reshape = lasagne.layers.ReshapeLayer(c_single, (1,30))

        qc_single = lasagne.layers.ConcatLayer([q_single,c_single_reshape]) # question is concatenated to each candidate
        print('qc_single',qc_single.output_shape)

        # regular embedding
        q_embedding_reshape = lasagne.layers.ReshapeLayer(q_embedding,(1,self.Q_MAX_LENGTH,self.EMBEDDING_DIM))
        c_embedding_individual = lasagne.layers.ReshapeLayer(c_embedding,(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH,self.EMBEDDING_DIM))
        c_embedding_combined = lasagne.layers.ReshapeLayer(c_embedding,(1,self.MAX_N_CANDIDATES*self.C_MAX_LENGTH,self.EMBEDDING_DIM))
        print('Question GRU Input shape',q_embedding_reshape.output_shape)

        qRep = lasagne.layers.GRULayer(q_embedding_reshape, Q_N_HIDDEN, grad_clipping=GRAD_CLIP, only_return_final=True)
        print('qRep output shape',qRep.output_shape)
        repeated_qRep = RepeatLayer(qRep,self.MAX_N_CANDIDATES)
        
        hidden_update = lasagne.layers.Gate(W_cell=None,nonlinearity=lasagne.nonlinearities.tanh)
        resetgate = lasagne.layers.Gate(W_cell=None)
        updategate = lasagne.layers.Gate(W_cell=None)

        c_combined_Rep_f = lasagne.layers.GRULayer(c_embedding_combined, 2*C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_combined_Rep_b = lasagne.layers.GRULayer(c_embedding_combined, 2*C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_combined_Rep = lasagne.layers.ElemwiseSumLayer([c_combined_Rep_f,c_combined_Rep_b])

        cRep_merge = lasagne.layers.ReshapeLayer(c_combined_Rep,(1,-1))
        qcRep_merge = lasagne.layers.ConcatLayer([cRep_merge,qc_single],axis=1) # question rep is concatenated to candidate rep 
        print('qcRep_merge',qcRep_merge.output_shape)
        qcRep_merge_single = lasagne.layers.ConcatLayer([qcRep_merge,qRep],axis=1)
        n_correct_prediction = lasagne.layers.DenseLayer(qcRep_merge_single, num_units=50, nonlinearity=lasagne.nonlinearities.tanh)
        n_correct_prediction_out = lasagne.layers.DenseLayer(n_correct_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        print('c_individual_Rep input shape',c_embedding_individual.output_shape)
        c_individual_Rep_f = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_individual_Rep_b = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_individual_Rep = lasagne.layers.ElemwiseSumLayer([c_individual_Rep_f,c_individual_Rep_b])

        print('c_individual_Rep output shape',c_individual_Rep.output_shape)

        qcRep = lasagne.layers.ConcatLayer([repeated_qRep,c_individual_Rep]) # question is concatenated to each candidate
        repeated_n_correct_prediction = RepeatLayer(n_correct_prediction_out,self.MAX_N_CANDIDATES) # contain probability also concatenated
        feature_merge = lasagne.layers.ConcatLayer([qcRep,repeated_n_correct_prediction]) # KEY MOVE
        print('feature_merge input', feature_merge.output_shape)

        feature_merge_1 = lasagne.layers.ConcatLayer([feature_merge,c_single])
        feature_merge_2 = lasagne.layers.ConcatLayer([feature_merge_1,repeated_q_single])
        print('feature_merge_1 input', feature_merge_2.output_shape)
        candidate_prediction = lasagne.layers.DenseLayer(feature_merge_2, num_units=50, nonlinearity=lasagne.nonlinearities.sigmoid)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        print('candidate_prediction_out output', candidate_prediction_out.output_shape)
        return [n_correct_prediction_out, candidate_prediction_out]

def main():
    lstm = Single('D_Single',3000,1,5)
    lstm.run()
if __name__ == '__main__':
    main()