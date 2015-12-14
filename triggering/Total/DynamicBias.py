from base import *
from lasagne.regularization import l2, l1
import lasagne

class DynamicBias(Base):
    def __init__(self,name,macro_batch_size,micro_batch_size,end_epoch):
        super(DynamicBias, self).__init__(name,macro_batch_size,micro_batch_size,end_epoch)
        self.learning_rate = 0.00015
        
        self.best_n_correct_loss = 100
        self.best_question_bias = 0
        self.best_question_f1 = 0
        
        self.best_question_margin_bias = 0
        self.best_question_margin_f1 = 0

        self.best_candidate_loss = 100
        self.best_candidate_bias = 0
        self.best_candidate_f1 = 0

        self.best_selected_candidate_f1 = 0

        self.best_n_correct_loss = 100
        self.best_n_correct_bias = 0
        self.best_n_correct_f1 = 0
        
        self.overfit = False

    def build_model(self):
        print('Building model')
        questions_in = T.imatrix('questions')
        candidates_in = T.imatrix('candidates')
        q_masks_in = T.matrix('q_masks')
        c_masks_in = T.matrix('c_masks')
        c_labels_in = T.fvector('candidate_labels')
        i = T.iscalar()

        outputs = self.define_layers(questions_in,candidates_in,q_masks_in,c_masks_in)

        self.model = lasagne.layers.get_all_layers(outputs)
        assert lasagne.layers.count_params(self.model) == lasagne.layers.count_params(outputs[1]), 'Wrong number of parameters for candidate output'

        train_n_correct_pred = T.clip(lasagne.layers.get_output(outputs[0], trainable=True), 1.0e-7, 1.0 - 1.0e-7)
        train_candidate_pred = T.clip(lasagne.layers.get_output(outputs[1], trainable=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)

        test_n_correct_pred = T.clip(lasagne.layers.get_output(outputs[0], deterministic=True), 1.0e-7, 1.0 - 1.0e-7)
        test_candidate_pred = T.clip(lasagne.layers.get_output(outputs[1], deterministic=True).flatten(), 1.0e-7, 1.0 - 1.0e-7)
        
        # theano.printing.pydotprint(test_existence, outfile="graph.png", var_with_name_simple=True)  

        self.question_params = lasagne.layers.get_all_params(outputs[0], trainable=True)
        self.candidate_params = lasagne.layers.get_all_params(outputs[1], trainable=True)

        train_n_correct_pred, train_n_correct_label, train_candidates_pred, train_candidate_label, train_n_correct_loss, train_candidate_loss, train_argmax_candidate, train_argmin_candidate = self.define_losses(train_n_correct_pred,train_candidate_pred,c_labels_in)
        test_n_correct_pred, test_n_correct_label, test_candidates_pred, test_candidate_label, test_n_correct_loss, test_candidate_loss, test_argmax_candidate, test_argmin_candidate = self.define_losses(test_n_correct_pred,test_candidate_pred,c_labels_in)
        
        print('learning rate {}'.format(self.learning_rate))
        print('{} parameters n_correct'.format(lasagne.layers.count_params(outputs[0])))
        print('{} parameters candidate'.format(lasagne.layers.count_params(outputs[1])))

        updates = dual_loss_adam(train_candidate_loss, train_n_correct_loss, self.candidate_params, self.question_params, learning_rate=self.learning_rate)
        # updates = OrderedDict() # disable optimizer to debug

        qb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype=np.int32)
        cb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype=np.int32)
        qmb = np.empty((self.macro_batch_size,1,self.Q_MAX_LENGTH), dtype=np.float32)
        cmb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES,self.C_MAX_LENGTH), dtype=np.float32)
        lb = np.empty((self.macro_batch_size,self.MAX_N_CANDIDATES), dtype=np.float32)

        self.questions_macro_batch = theano.shared(qb, borrow=True)
        self.candidates_macro_batch = theano.shared(cb, borrow=True)
        self.q_masks_macro_batch = theano.shared(qmb, borrow=True)
        self.c_masks_macro_batch = theano.shared(cmb, borrow=True)
        self.c_labels_macro_batch = theano.shared(lb, borrow=True)

        self.train_fn = theano.function([i], [train_n_correct_pred, train_n_correct_label, train_candidates_pred, train_candidate_label, train_n_correct_loss, train_candidate_loss, train_argmax_candidate, train_argmin_candidate], updates=updates,
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            c_labels_in: self.c_labels_macro_batch[i]
        })

        self.test_fn = theano.function([i], [test_n_correct_pred, test_n_correct_label, test_candidates_pred, test_candidate_label, test_n_correct_loss, test_candidate_loss, test_argmax_candidate, test_argmin_candidate],
            givens={
            questions_in: self.questions_macro_batch[i],
            candidates_in: self.candidates_macro_batch[i],
            q_masks_in: self.q_masks_macro_batch[i],
            c_masks_in: self.c_masks_macro_batch[i],
            c_labels_in: self.c_labels_macro_batch[i]
        })

        # pred = T.ivector()
        # label = T.ivector()
        # model_TP = (pred & label).nonzero_values().shape[0]
        # model_FP = pred.nonzero_values().shape[0]-model_TP
        # model_prec = model_TP/(model_TP + model_FP)
        # model_FN = (~pred & label).nonzero_values().shape[0]
        # model_rec = model_TP/(model_TP+model_FN)
        # model_f1 = 2*model_prec*model_rec/(model_prec+model_rec)
        # self.eval = theano.function([pred,label], [model_prec,model_rec,model_f1], allow_input_downcast=True)

    def define_losses(self,n_correct_pred,candidate_pred,candidate_label):

        nonzeros = candidate_label.nonzero()
        ONE = T.cast(T.ones_like(candidate_label),'float32')
        TRIPLE = T.inc_subtensor(ONE[nonzeros],np.float32(2))

        n_correct = T.cast(candidate_label.nonzero_values().shape[0],'float32') # number of correct answers
        n_correct_pred = T.switch(T.eq(T.argmax(n_correct_pred),np.float32(0)),np.float32(0),np.float32(1))
        n_correct_pred = T.switch(T.gt(T.argmax(n_correct_pred),np.float32(1)),np.float32(0),n_correct_pred)
        n_correct_loss = (((n_correct - n_correct_pred)/3)**2).mean()
        # n_correct_loss = ((((n_correct - n_correct_pred)/3)**2)*2).mean() # cost senstivity without effect

        candidate_loss = (lasagne.objectives.binary_crossentropy(candidate_pred, candidate_label)*TRIPLE).mean()/np.float32(2)

        argmax_candidate = T.argmax(candidate_pred)
        argmin_candidate = T.argmin(candidate_pred)
        return n_correct_pred, n_correct, candidate_pred, candidate_label, n_correct_loss, candidate_loss, argmax_candidate, argmin_candidate

    def train_model(self):
        print('Starting Train')
        for epoch in xrange(self.start_epoch,self.end_epoch):
            start_time = time.time()
            self.epoch = epoch
            for i,split in enumerate(['train','dev']):
                self.process(i,split)
            print('Epoch: {} {} sec\n'.format(self.epoch+1,int(time.time()-start_time)))

    def evaluate_model(self):
        print('Starting Test')
        start_time = time.time()        
        self.process(2,'test')
        print('{} sec'.format(int(time.time()-start_time)))

    def process(self,i,split):
        candidate_loss = 0
        n_correct_loss = 0
        candidate_preds = []
        n_correct_pred = []
        candidate_label = []
        n_correct_label = []          
        argmax_candidate = []
        argmin_candidate = []

        macro_batch_count = self.questions[i].shape[0] // self.macro_batch_size
        micro_batch_count = self.macro_batch_size // self.micro_batch_size
        shuffled_micro_batches = range(macro_batch_count)
        random.shuffle(shuffled_micro_batches)
        for macro_batch_index in shuffled_micro_batches:
            self.set_all(i, macro_batch_index)
            shuffled_micro_batches = range(micro_batch_count)
            random.shuffle(shuffled_micro_batches)
            for micro_batch_index in shuffled_micro_batches:
                if split == 'train':
                    micro_n_correct_pred,micro_n_correct,micro_candidate_pred,micro_candidate,micro_n_correct_loss,micro_candidate_loss,micro_question_argmax,micro_question_argmin = self.train_fn(micro_batch_index)
                else:
                    micro_n_correct_pred,micro_n_correct,micro_candidate_pred,micro_candidate,micro_n_correct_loss,micro_candidate_loss,micro_question_argmax,micro_question_argmin = self.test_fn(micro_batch_index)
                candidate_loss += micro_candidate_loss
                n_correct_loss += micro_n_correct_loss
                n_correct_pred.append(micro_n_correct_pred)
                candidate_preds.append(micro_candidate_pred)
                candidate_label.extend(micro_candidate)
                n_correct_label.append(micro_n_correct)
                argmax_candidate.append(micro_question_argmax+micro_batch_index*self.micro_batch_size)
                argmin_candidate.append(micro_question_argmin+micro_batch_index*self.micro_batch_size)

        if self.questions[i].shape[0] % self.macro_batch_size != 0:
            remainder = self.questions[i].shape[0]%self.macro_batch_size
            self.set_all_rest(i, remainder)
            shuffled_micro_batches = range(remainder)
            random.shuffle(shuffled_micro_batches)
            for micro_batch_index in shuffled_micro_batches:
                if split == 'train':
                    micro_n_correct_pred,micro_n_correct,micro_candidate_pred,micro_candidate,micro_n_correct_loss,micro_candidate_loss,micro_question_argmax,micro_question_argmin = self.train_fn(micro_batch_index)
                else:
                    micro_n_correct_pred,micro_n_correct,micro_candidate_pred,micro_candidate,micro_n_correct_loss,micro_candidate_loss,micro_question_argmax,micro_question_argmin = self.test_fn(micro_batch_index)
                candidate_loss += micro_candidate_loss
                n_correct_loss += micro_n_correct_loss
                n_correct_pred.append(micro_n_correct_pred)
                candidate_preds.append(micro_candidate_pred)
                candidate_label.extend(micro_candidate)
                n_correct_label.append(micro_n_correct)
                argmax_candidate.append(micro_question_argmax+micro_batch_index*self.micro_batch_size)
                argmin_candidate.append(micro_question_argmin+micro_batch_index*self.micro_batch_size)

            n_correct_loss/=self.questions[i].shape[0]
            candidate_loss/=self.questions[i].shape[0]
            
        self.evaluate(split,n_correct_loss,n_correct_pred,np.asarray(n_correct_label),candidate_loss,candidate_preds,np.asarray(candidate_label),argmax_candidate,argmin_candidate)


    def evaluate(self,split,n_correct_loss,n_correct_pred,n_correct_label,candidate_loss,candidate_preds,candidate_label,argmax_candidate,argmin_candidate):
        test = False
        if split is not 'train':
            test = True
        corrected_n_correct_pred, n_correct_bias = self.find_best_threshold(np.asarray(n_correct_pred),n_correct_label,'n_correct',test,'micro')
        n_correct_prec, n_correct_rec, n_correct_f1 = self.micro_score(corrected_n_correct_pred,n_correct_label)

        # n_correct_prec, n_correct_rec, n_correct_f1 = self.eval(corrected_n_correct_pred,n_correct_label) # core dumped float point error?
        # print('ncorrect',n_correct_prec,n_correct_rec,n_correct_f1)
        # --------------------------- choose top n_correct -----------------------------
        selected_candidate_preds = []
        for n_correct,candidate_pred in zip(corrected_n_correct_pred,candidate_preds):
            n_correct_idx = np.argpartition(candidate_pred, -n_correct)[-n_correct:]
            n_correct_candidate_pred = np.zeros(self.MAX_N_CANDIDATES)
            n_correct_candidate_pred[n_correct_idx] = 1
            selected_candidate_preds.extend(n_correct_candidate_pred)

        selected_candidate_prec, selected_candidate_rec, selected_candidate_f1 = self.score(selected_candidate_preds,candidate_label)

        # ----------------------------- choose threshold -------------------------------
        candidate_preds_flatten = np.asarray(candidate_preds).flatten()
        corrected_candidate_pred, candidate_bias = self.find_best_threshold(candidate_preds_flatten,candidate_label,'candidate',test,'binary')
        candidate_prec, candidate_rec, candidate_f1 = self.score(corrected_candidate_pred,candidate_label)

        # ----------------------------- question level margin ---------------------------------
        # does it make sense? many negative examples, not max margin classifier?
        question_pred_margin = np.zeros(len(argmax_candidate))
        question_label_margin = np.zeros(len(argmax_candidate))
        threshold = candidate_bias+.5
        for i,pred in enumerate(zip(candidate_preds_flatten[argmax_candidate], candidate_preds_flatten[argmin_candidate])):
            max_margin = abs(threshold-pred[0])
            min_margin = abs(threshold-pred[1])
            if max_margin < min_margin:
                question_pred_margin[i] = pred[0]
                question_label_margin[i] = candidate_label[argmax_candidate[i]]
            else:
                question_pred_margin[i] = pred[1]
                question_label_margin[i] = candidate_label[argmin_candidate[i]]
        corrected_question_pred_margin,question_margin_bias = self.find_best_threshold(question_pred_margin,question_label_margin,'question_margin',test,'binary')
        question_margin_prec, question_margin_rec, question_margin_f1 = self.score(corrected_question_pred_margin,question_label_margin)

        # ----------------------------- question level highest --------------------------------
        question_pred = candidate_preds_flatten[argmax_candidate]
        question_label = candidate_label[argmax_candidate]
        corrected_question_pred, question_bias = self.find_best_threshold(question_pred,question_label,'question',test,'binary')
        question_prec, question_rec, question_f1 = self.score(corrected_question_pred,question_label)

        self.print_scores(split, n_correct_loss,
            n_correct_prec,n_correct_rec,n_correct_f1,n_correct_bias,
            candidate_loss,candidate_prec,candidate_rec,candidate_f1,candidate_bias,
            selected_candidate_prec, selected_candidate_rec, selected_candidate_f1,
            question_prec,question_rec,question_f1,question_bias,
            question_margin_prec,question_margin_rec,question_margin_f1,question_margin_bias)
        if split == 'train':
            self.save_thresholds(n_correct_bias,candidate_bias,question_bias,question_margin_bias)
        if split == 'dev':
            self.save_best(n_correct_loss, n_correct_f1, n_correct_bias, candidate_loss, candidate_f1, candidate_bias, selected_candidate_f1, question_f1, question_bias, question_margin_f1, question_margin_bias)


    def find_best_threshold(self,scores,labels,measure,test,average='binary'):
        best_f1 = 0
        best_bias = 0
        if test==True:
            if measure == 'question':
                best_pred = np.rint(scores+self.best_question_bias)
                best_pred[best_pred > 1] = 1
                best_bias = self.best_question_bias
            if measure == 'question_margin':
                best_pred = np.rint(scores+self.best_question_margin_bias)
                best_pred[best_pred > 1] = 1
                best_bias = self.best_question_margin_bias            
            elif (measure == 'candidate'):
                best_pred = np.rint(scores+self.best_candidate_bias)
                best_pred[best_pred > 1] = 1
                best_bias = self.best_candidate_bias
            elif (measure == 'n_correct'):
                best_pred = np.rint(scores+self.best_n_correct_bias)
                best_pred[best_pred > 1] = 1
                best_bias = self.best_n_correct_bias
        else:
            for bias in np.arange(0,1,0.05):
                pred = np.around(scores+bias)
                prec = precision_score(labels,pred,average=average)
                rec = recall_score(labels,pred,average=average)
                f1 = 2*prec*rec/(prec+rec)
                if f1 > best_f1:
                    best_f1 = f1
                    best_pred = pred
                    best_bias = bias
        return best_pred.astype(np.int32,copy=False),best_bias

    def save_thresholds(self,n_correct_bias,candidate_bias,question_bias,question_margin_bias):
        if not self.overfit:
            self.best_n_correct_bias = n_correct_bias
            self.best_candidate_bias = candidate_bias
            self.best_question_bias = question_bias
            self.best_question_margin_bias = question_margin_bias

    def save_best(self,n_correct_loss, n_correct_f1, n_correct_bias, candidate_loss, candidate_f1, candidate_bias, selected_candidate_f1, question_f1, question_bias, question_margin_f1, question_margin_bias):
        if candidate_loss < self.best_candidate_loss:
            self.overfit = False
            self.best_n_correct_f1 = n_correct_f1
            self.best_candidate_f1 = candidate_f1
            self.best_selected_candidate_f1 = selected_candidate_f1
            self.best_question_f1 = question_f1
            self.best_question_margin_f1 = question_margin_f1
            self.best_n_correct_loss = n_correct_loss
            self.best_candidate_loss = candidate_loss
            self.write_model_data(n_correct_loss, n_correct_f1, n_correct_bias, candidate_loss, candidate_f1, candidate_bias, question_f1, question_bias)
        else:
            self.overfit = True

    def read_model_data(self):
        if self.load == False:
            return
        print('Loading Model')
        modelFile = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
        if os.path.isfile(modelFile):
            with open(modelFile, 'rb') as f:
                self.start_epoch,self.best_candidate_f1,self.best_candidate_bias,self.best_question_f1,self.best_question_bias,self.best_candidate_loss,params = pickle.load(f)
            self.end_epoch += self.start_epoch
            lasagne.layers.set_all_param_values(self.model,params)

    def write_model_data(self,n_correct_loss, n_correct_f1, n_correct_bias, candidate_loss, candidate_f1, candidate_bias, question_f1, question_bias):
        if self.save == False:
            return
        filename = os.path.join('./models/', '%s.%s' % (self.name, 'params'))
        with open(filename, 'wb') as f:
            pickle.dump((self.epoch,self.best_candidate_f1,self.best_candidate_bias,self.best_question_f1,self.best_question_bias,self.best_candidate_loss,lasagne.layers.get_all_param_values(self.model)), f, protocol=-1)

    def print_scores(self,split,
        n_correct_loss,n_correct_prec,n_correct_rec,n_correct_f1,n_correct_bias,
        candidate_loss,candidate_prec,candidate_rec,candidate_f1,candidate_bias,
        selected_candidate_prec, selected_candidate_rec, selected_candidate_f1,
        question_prec,question_rec,question_f1,question_bias,
        question_margin_prec,question_margin_rec,question_margin_f1,question_margin_bias):
        n_string = '{} n_correct L: {:.4f} P: {:.2f} R: {:.2f} F1: {:.2f} Bias: {:.2f}'.format(split, n_correct_loss,100*n_correct_prec,100*n_correct_rec,100*n_correct_f1, n_correct_bias)
        c_string = '{} candidate L: {:.4f} P: {:.2f} R: {:.2f} F1: {:.2f} Bias: {:.2f}'.format(split, candidate_loss,100*candidate_prec,100*candidate_rec,100*candidate_f1, candidate_bias)
        sc_string = '{} candidate-selection L: {:.4f} P: {:.2f} R: {:.2f} F1: {:.2f}'.format(split, candidate_loss,100*selected_candidate_prec,100*selected_candidate_rec,100*selected_candidate_f1)
        q_string = '{} question P: {:.2f} R: {:.2f} F1: {:.2f} Bias: {:.2f}'.format(split, 100*question_prec,100*question_rec,100*question_f1, question_bias)
        qm_string = '{} question-margin P: {:.2f} R: {:.2f} F1: {:.2f} Bias: {:.2f}'.format(split, 100*question_margin_prec,100*question_margin_rec,100*question_margin_f1, question_margin_bias)
        print(n_string)
        print(c_string)
        print(sc_string)
        print(q_string)
        print(qm_string)

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

        # regular embedding
        q_embedding_reshape = lasagne.layers.ReshapeLayer(q_embedding,(1,self.Q_MAX_LENGTH,self.EMBEDDING_DIM))
        c_embedding_individual = lasagne.layers.ReshapeLayer(c_embedding,(self.MAX_N_CANDIDATES,self.C_MAX_LENGTH,self.EMBEDDING_DIM))
        c_embedding_combined = lasagne.layers.ReshapeLayer(c_embedding,(1,self.MAX_N_CANDIDATES*self.C_MAX_LENGTH,self.EMBEDDING_DIM))
        print('Question GRU Input shape',q_embedding_reshape.output_shape)

        '''
        # subspace
        q_subspace = lasagne.layers.DenseLayer(q_embedding,self.Q_MAX_LENGTH*SUBSPACE_DIM, nonlinearity=lasagne.nonlinearities.tanh)
        c_subspace = lasagne.layers.DenseLayer(c_embedding,self.C_MAX_LENGTH*SUBSPACE_DIM, nonlinearity=lasagne.nonlinearities.tanh)
        print('q_subspace',q_subspace.output_shape)
        print('c_subspace',c_subspace.output_shape)
        q_embedding_reshape = lasagne.layers.ReshapeLayer(q_subspace,(1,self.Q_MAX_LENGTH,SUBSPACE_DIM))
        c_embedding_individual = lasagne.layers.ReshapeLayer(c_subspace,(self.n_correct_CANDIDATES,self.C_MAX_LENGTH,SUBSPACE_DIM))
        c_embedding_combined = lasagne.layers.ReshapeLayer(c_subspace,(1,self.n_correct_CANDIDATES*self.C_MAX_LENGTH,SUBSPACE_DIM))
        print('Question GRU Input shape',q_embedding_reshape.output_shape)
        '''
        
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
        qcRep_merge = lasagne.layers.ConcatLayer([cRep_merge,qRep],axis=1) # question rep is concatenated to candidate rep 

        n_correct_prediction = lasagne.layers.DenseLayer(qcRep_merge, num_units=50, nonlinearity=lasagne.nonlinearities.tanh)
        n_correct_prediction_out = lasagne.layers.DenseLayer(n_correct_prediction, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)
        print('n_correct prediction out output shape',n_correct_prediction_out.output_shape)
        print('c_individual_Rep input shape',c_embedding_individual.output_shape)
        c_individual_Rep_f = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True)
        c_individual_Rep_b = lasagne.layers.GRULayer(c_embedding_individual, C_N_HIDDEN, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, gradient_steps=GRAD_STEPS, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        c_individual_Rep = lasagne.layers.ElemwiseSumLayer([c_individual_Rep_f,c_individual_Rep_b])

        print('c_individual_Rep output shape',c_individual_Rep.output_shape)

        qcRep = lasagne.layers.ConcatLayer([repeated_qRep,c_individual_Rep]) # question is concatenated to each candidate
        
        repeated_n_correct_prediction = RepeatLayer(n_correct_prediction_out,self.MAX_N_CANDIDATES) # contain probability also concatenated

        feature_merge = lasagne.layers.ConcatLayer([qcRep,repeated_n_correct_prediction])
        print('candidate_prediction input', feature_merge.output_shape)
        candidate_prediction = lasagne.layers.DenseLayer(feature_merge, num_units=50, nonlinearity=lasagne.nonlinearities.sigmoid)
        candidate_prediction_out = lasagne.layers.DenseLayer(candidate_prediction, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        print('candidate_prediction_out output', candidate_prediction_out.output_shape)
        return [n_correct_prediction_out, candidate_prediction_out]
def main():
    lstm = DynamicBias('DynamicBias',3000,1,5)
    lstm.run()
if __name__ == '__main__':
    main()