from keras.layers.convolutional import *
from keras.layers.core import *
from keras.models import Sequential
from numpy import array
import itertools
import cPickle as pickle
from pprint import pprint
from keras.utils import generic_utils
from generate_input_file import load_questions_from_file
from decomposition.sentence import decompose_questions
from sklearn.linear_model import LogisticRegression

import globals
import extras

if globals.options.nb_epoch is not None:
    nb_epoch = int(globals.options.nb_epoch)
else:
    nb_epoch = globals.nb_epoch

if globals.options.batch_size is not None:
    batch_size = int(globals.options.batch_size)
else:
    batch_size = globals.batch_size

train_data = None
validate_data = None
test_data = None
q_limit = {}


def load_data(path, feature_file, label_file):
    globals.logger.info("Loading data from %s..." % path)
    X = array(pickle.load(open(path + feature_file)))
    y = array(pickle.load(open(path + label_file)))
    globals.logger.info("Data loaded.")

    return X, y


def dump_data(path, data, feature_file, label_file):
    if 'labels' not in data or 'features' not in data:
        raise ValueError("'data' is not a proper form.")

    globals.logger.info("Dumping data to %s..." % path)
    pickle.dump(data['features'], open(path + feature_file, "wb"))
    pickle.dump(data['labels'], open(path + label_file, "wb"))
    globals.logger.info("Data dumped.")


def test_nn(generate_logistic_data=False):
    # Load data for NN
    X_train, y_train = load_data(train_data, globals.nn_features_file, globals.nn_labels_file)
    X_validate, y_validate = load_data(validate_data, globals.nn_features_file, globals.nn_labels_file)
    X_test, y_test = load_data(test_data, globals.nn_features_file, globals.nn_labels_file)

    # Train section on the training set
    model = train(X_train, y_train, X_validate, y_validate)

    # Test on the validation set
    globals.logger.info("Testing NN model on validation set...")
    results = test_model(model, X_validate, y_validate)
    globals.logger.info("Model tested on validation set.")

    # Try to find a threshold on the validation set
    globals.logger.info("Finding best threshold on the validation set...")
    best_threshold = find_threshold(y_validate, results["y_predicted_scores"])

    # Validate the threshold on the validation set
    precision, recall, f1 = evaluate_with_threshold(y_validate, results["y_predicted_scores"], best_threshold)
    globals.logger.info("Found threshold: %f. Precision/recall/f1 on found threshold "
                        "over validation set: %f/%f/%f" % (best_threshold, precision, recall, f1))

    # Perform a test on the test set
    globals.logger.info("Testing NN on test set and evaluating using the threshold...")
    results = test_model(model, X_test, y_test)
    precision, recall, f1 = evaluate_with_threshold(y_test, results["y_predicted_scores"], best_threshold)
    globals.logger.info("Precision/recall/f1 on best_threshold threshold over test set: %f/%f/%f" %
                        (precision, recall, f1))

    if generate_logistic_data is False:
        return

    # Load data for Logistic Regression files generation
    questions_train = load_questions_from_file("train", q_limit['train'])
    questions_validate = load_questions_from_file("validate", q_limit['validate'])
    questions_test = load_questions_from_file("test", q_limit['test'])

    # Training set generation
    globals.logger.info("Generating the training set for Logistic Regression")
    samples = decompose_questions(questions_train, stemming=True)
    dict_nn = test_model(model, X_train, y_train)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(train_data, {'features': samples, 'labels': y_train}, globals.lr_features_file, globals.lr_labels_file)

    # Validation set generation
    globals.logger.info("Generating the validation set for Logistic Regression")
    samples = decompose_questions(questions_validate, stemming=True)

    dict_nn = test_model(model, X_validate, y_validate)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(validate_data, {'features': samples, 'labels': y_validate},
              globals.lr_features_file, globals.lr_labels_file)

    # Test section
    globals.logger.info("Generating the test set for Logistic Regression")
    samples = decompose_questions(questions_test, stemming=True)

    dict_nn = test_model(model, X_test, y_test)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(test_data, {'features': samples, 'labels': y_test}, globals.lr_features_file, globals.lr_labels_file)


def generate_train_logistic():
    # Load data for Logistic Regression files generation
    questions_train = load_questions_from_file("train", q_limit['train'])
    questions_validate = load_questions_from_file("validate", q_limit['validate'])
    questions_test = load_questions_from_file("test", q_limit['test'])

    # Load data for NN
    X_train, y_train = load_data(train_data, globals.nn_features_file, globals.nn_labels_file)
    X_validate, y_validate = load_data(validate_data, globals.nn_features_file, globals.nn_labels_file)
    X_test, y_test = load_data(test_data, globals.nn_features_file, globals.nn_labels_file)

    # Train section on the training set
    globals.logger.info("Training NN on the training set...")
    model = train(X_train, y_train, X_validate, y_validate)
    globals.logger.info("Training NN done.")

    # Test on the validation set
    globals.logger.info("Testing NN model on validation set...")
    results = test_model(model, X_validate, y_validate)
    globals.logger.info("Model tested on validation set.")

    # Try to find a threshold on the validation set
    globals.logger.info("Finding best threshold for validation set...")
    best_threshold = find_threshold(y_validate, results["y_predicted_scores"])

    samples = decompose_questions(questions_train, stemming=True)

    dict_nn = test_model(model, X_train, y_train)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(train_data, {'features': samples, 'labels': y_train})

    # Validate section
    samples = decompose_questions(questions_validate, stemming=True)

    dict_nn = test_model(model, X_validate, y_validate)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(validate_data, {'features': samples, 'labels': y_validate})

    # Test section
    samples = decompose_questions(questions_test, stemming=True)

    dict_nn = test_model(model, X_test, y_test)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(test_data, {'features': samples, 'labels': y_test})


def test_lr():
    #questions_train = load_questions_from_file("train", q_limit['train'])
    #questions_test = load_questions_from_file("test", q_limit['test'])
    #samples_train = decompose_questions(questions_train, stemming=True)
    #samples_test = decompose_questions(questions_test, stemming=True)

    # Load the data
    X_train, y_train = load_data(train_data, globals.lr_features_file, globals.lr_labels_file)
    #X_validate, y_validate = load_data(validate_data, globals.lr_features_file, globals.lr_labels_file)
    X_test, y_test = load_data(test_data, globals.lr_features_file, globals.lr_labels_file)

    y_train_flatten = list(itertools.chain(*y_train))
    #y_validate_flatten = list(itertools.chain(*y_validate))
    y_test_flatten = list(itertools.chain(*y_test))

    lr = LogisticRegression()
    lr.fit(X_train, y_train_flatten)

    predictions = lr.decision_function(X_test)
    best_threshold = find_threshold(y_test, predictions)

    precision, recall, f1 = evaluate_with_threshold(y_test, predictions, best_threshold)
    globals.logger.info("Found threshold: %f. Precision/recall/f1 on found threshold "
                        "over test set: %f/%f/%f" % (best_threshold, precision, recall, f1))

    predictions = lr.decision_function(X_train)
    precision, recall, f1 = evaluate_with_threshold(y_train, predictions, best_threshold)
    globals.logger.info("Precision/recall/f1 on found threshold "
                        "over training set: %f/%f/%f" % (precision, recall, f1))

    #best_f1 = 0.0


def test_idf():
    questions_train = load_questions_from_file("train", q_limit['train'])
    X_train, y_train = load_data(train_data, globals.lr_features_file, globals.lr_labels_file)
    y_train_flatten = list(itertools.chain(*y_train))

    samples = decompose_questions(questions_train, stemming=True)

    print("Returned: ")
    i = 0
    for q_set in samples:
        print("Question starts:")
        for sample in q_set:
            l_part = ("Sample: " + str(sample)).ljust(35)
            if y_train_flatten[i] == 1:
                print("%s class label: %d" % (l_part, y_train_flatten[i]))
            else:
                print("%s" % l_part)
            i += 1


def test_model(model, X_test, y_test):
    # Get test loss and test accuracy
    (test_loss, test_accuracy) = model.evaluate(X_test, list(itertools.chain(*y_test)), show_accuracy=True, batch_size=1)
    #print("test loss: %.4f, test accuracy: %.4f" % (test_loss, test_accuracy))

    # Get predicted classes and scores
    # TODO: Classes probably not needed
    #y_predicted_classes = model.predict_classes(X_test, batch_size=1)
    y_predicted_scores = model.predict(X_test, batch_size=1)
    #print("y_gold: %s\n\ny_predicted: %s" % (y_test, y_predicted_classes))

    return {"test_loss": test_loss,
            "test_accuracy": test_accuracy,
            #"y_predicted_classes": y_predicted_classes,
            "y_predicted_scores": y_predicted_scores}


def train(X_train, y_train, X_validate, y_validate, show_accuracy=True):
    model = Sequential()
    model.add(Convolution2D(globals.dimension, 1, 2, globals.dimension))
    model.add(Activation('tanh'))

    model.add(extras.AveragePooling2D(poolsize=(globals.s_size,1)))

    model.add(Flatten())

    model.add(Dense(2*globals.dimension, 1))

    model.add(Activation('sigmoid'))

    globals.logger.info("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
    globals.logger.info("Model compiled, start training...")

    y_train_flatted = list(itertools.chain(*y_train))
    y_validate_flatted = list(itertools.chain(*y_validate))
    nb_batch = len(X_train)/batch_size + 1

    best_loss = 1.0
    best_loss_index = -1
    best_f1 = 0.0
    best_f1_index = 0
    previous_loss = 0

    #model.fit(X_train, y_train_flatted, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2)

    for e in range(nb_epoch):
        print("Epoch %d" % e)
        progress_bar = generic_utils.Progbar(X_train.shape[0])
        for i in range(nb_batch):
            train_loss,train_accuracy = model.train_on_batch(X_train[i*batch_size:(i+1)*batch_size], y_train_flatted[i*batch_size:(i+1)*batch_size],accuracy=show_accuracy)
            progress_bar.add(batch_size, values=[("train loss", train_loss),("train accuracy:", train_accuracy)])

        # Check the scores after this batch
        results = test_model(model, X_validate, y_validate)
        best_threshold = find_threshold(y_validate, results["y_predicted_scores"])
        precision, recall, f1 = evaluate_with_threshold(y_validate, results["y_predicted_scores"], best_threshold)
        globals.logger.info("OVER Validation found threshold: %.4f.\n"
                            "Validation loss: %.4f - Validation accuracy: %.4f\n"
                            "Precision: %.4f - Recall: %.4f - F1: %.4f" % (best_threshold,
                                                                           results['test_loss'], results['test_accuracy'],
                                                                            precision, recall, f1))

        if e == 1:
            previous_loss = results['test_loss']
        if f1 >= best_f1 and e > 1 and \
                        float(results['test_loss']) < float(previous_loss) and \
                                float(previous_loss)-float(results['test_loss']) <= 0.0003:
            # Better score
            best_f1 = f1
            best_f1_index = e
            model.save_weights("model_weights.h5", overwrite=True)

        previous_loss = results['test_loss']

        #val_loss,val_accuracy = model.evaluate(X_validate, y_validate_flatted, batch_size=1, show_accuracy=show_accuracy)
        #print("val_loss: %.4f - val_acc: %.4f" % (val_loss, val_accuracy))
        #if best_loss > val_loss:
        #    # Try threshold
        #    results = test_model(model, X_validate, y_validate)
        #    best_threshold = find_threshold(y_validate, results["y_predicted_scores"])
        #    precision, recall, f1 = evaluate_with_threshold(y_validate, results["y_predicted_scores"], best_threshold)
        #    globals.logger.info("Found threshold: %f. Precision/recall/f1 on found threshold "
        #                        "over validation set: %f/%f/%f" % (best_threshold, precision, recall, f1))

        #    best_loss = val_loss
        #    best_loss_index = e
        #    model.save_weights("model_weights.h5", overwrite=True)

    #globals.logger.info("Training done, best loss on validation is: %.4f for epoch nr: %d" %
    #                    (best_loss, best_loss_index))
    globals.logger.info("Training done, best f1 on validation is: %.4f for epoch nr: %d" %
                       (best_f1, best_f1_index))
    model.load_weights("model_weights.h5")

    # aaa = model.fit(X_train, list(itertools.chain(*y_train)),
    #           nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True,
    #           validation_data=validation_data)

    # store the model and weights
    # json_model = model.to_json()
    # file_prefix = globals.models_path + globals.mode + "." + str(globals.q_limit) if globals.q_limit > -1 else "all"
    #
    # with open(file_prefix + ".model", 'w') as outfile:
    #     json.dump(json_model, outfile)
    # model.save_weights(file_prefix + ".weights", overwrite=True)

    return model


def find_threshold(questions_gold_sets, predictions):
    # Make a pass through all corrects and calculate their average
    index_begin = 0
    sum_of_scores = 0.0
    all_questions_number = 0

    for question_set in questions_gold_sets:
        index_end = index_begin + len(question_set)
        if 1 in question_set:
            gold_answer_id = question_set.index(1)
        else:
            gold_answer_id = -1

        predictions_slice = predictions[index_begin:index_end].flatten()

        if gold_answer_id > -1:
            sum_of_scores += predictions_slice[gold_answer_id]
            all_questions_number += 1

        index_begin = index_end

    average_threshold = sum_of_scores/all_questions_number
    globals.logger.info("Found initial average threshold of: %f. Now searching best threshold..." % average_threshold)
    avg_precision, avg_recall, avg_f1 = evaluate_with_threshold(questions_gold_sets, predictions, average_threshold)

    # If average f1 is 0.0, there is no need to search for
    if avg_f1 == 0.0:
        return average_threshold

    delta = 0.00001

    # Go incrementing threshold
    current_threshold = average_threshold
    up_f1 = avg_f1
    up_threshold = current_threshold
    while True:
        current_threshold += delta
        precision, recall, f1 = evaluate_with_threshold(questions_gold_sets, predictions, current_threshold)
        if f1 > up_f1:
            up_f1 = f1
            up_threshold = current_threshold
        elif f1 < up_f1:
            break

    # Go decrementing threshold
    current_threshold = average_threshold
    down_f1 = avg_f1
    down_threshold = current_threshold
    while True:
        current_threshold -= delta
        precision, recall, f1 = evaluate_with_threshold(questions_gold_sets, predictions, current_threshold)
        if f1 > down_f1:
            down_f1 = f1
            down_threshold = current_threshold
        elif f1 < down_f1:
            break

    print("After search. \nup_f1, up_threshold: %f, %f\ndown_f1, down_threshold: %f, %f" %
          (up_f1, up_threshold, down_f1, down_threshold))

    if up_f1 > avg_f1 or down_f1 > avg_f1:
        if up_f1 > down_f1:
            return up_threshold
        elif up_f1 == down_f1:
            return down_threshold
        else:
            return down_threshold
    else:
        return average_threshold


def evaluate_with_threshold(questions_gold_sets, predictions, threshold):
    index_begin = 0
    all_questions_with_answers = 0
    predicted_questions = 0
    correctly_predicted_questions = 0

    for question_set in questions_gold_sets:
        # Get the slice from predictions for this question_set
        index_end = index_begin + len(question_set)
        predictions_slice = predictions[index_begin:index_end]

        # Find the maximum value and if it's > threshold,
        # find its index in the list
        max_val = max(predictions_slice)
        if max_val > threshold:
            predicted_answer = [i for i, j in enumerate(predictions_slice) if j == max_val]
        else:
            predicted_answer = [-1]

        # Check if this question has an answer.
        # -1 if not, index in its question_set if yes
        if 1 in question_set:
            gold_answer_id = question_set.index(1)
        else:
            gold_answer_id = -1

        # If there is an answer, increment number of all questions
        if gold_answer_id > -1:
            all_questions_with_answers += 1

        # If the question predicted with the answer, increment predicted_questions
        if predicted_answer[0] > -1:
            predicted_questions += 1

        # If the question predicted correctly, increment correctly_predicted_questions
        if predicted_answer[0] > -1 and predicted_answer[0] == gold_answer_id:
            correctly_predicted_questions += 1

        index_begin = index_end

    # Calculate precision, recall and F1
    if correctly_predicted_questions == 0 or predicted_questions == 0:
        return 0.0, 0.0, 0.0
    precision = float(correctly_predicted_questions)/predicted_questions
    recall = float(correctly_predicted_questions)/all_questions_with_answers
    f1 = (2*precision*recall)/(precision+recall) if (precision > 0.0 and recall > 0.0) else 0.0

    return precision, recall, f1


def get_config():
    config = {"train_data": train_data,
              "validate_data": validate_data,
              "test_data": test_data,
              "exp_mode": globals.exp_mode,
              "nb_epoch": nb_epoch,
              "batch_size": batch_size}
    return config


if __name__ == "__main__":
    if globals.exp_mode is None:
        raise ValueError("Run with experiment mode (-e)")

    if not globals.options.train_data:
        raise ValueError("Missing training data, use --train")
    else:
        train_data = globals.options.train_data

    if not globals.options.validate_data:
        raise ValueError("Missing training data, use --validate")
    else:
        validate_data = globals.options.validate_data

    if not globals.options.test_data:
        raise ValueError("Missing training data, use --test")
    else:
        test_data = globals.options.test_data

    q_limit['train'] = int(''.join(x for x in train_data if x.isdigit())) if any(i.isdigit() for i in train_data) \
        else "all"
    q_limit['validate'] = int(''.join(x for x in validate_data if x.isdigit())) if any(i.isdigit() for i in
                                                                                       validate_data) else "all"
    q_limit['test'] = int(''.join(x for x in test_data if x.isdigit())) if any(i.isdigit() for i in test_data) \
        else "all"

    print("exp options")
    pprint(get_config())

    if globals.exp_mode == "test_nn":
        test_nn()
    elif globals.exp_mode == "generate_train_logistic":
        test_nn(generate_logistic_data=True)
    elif globals.exp_mode == "test_idf":
        test_idf()
    elif globals.exp_mode == "test_lr":
        test_lr()
    else:
        raise RuntimeError("Unknown exp_mode option.")


# X = array(pickle.load(open('data/train.130/features')))
# y = array(pickle.load(open('data/train.130/labels')))
# model = Sequential()
# model.add(Convolution2D(globals.dimension, 1, 2, globals.dimension))
# model.add(Activation('tanh'))
# model.add(AveragePooling2D(poolsize=(globals.s_size,1)))
# model.add(Flatten())
# model.add(Dense(2*globals.dimension,1))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
#
# get_1st_layer_output = theano.function([model.layers[0].input],
#                                         model.layers[0].get_output(train=False))
# get_2nd_layer_output = theano.function([model.layers[0].input],
#                                        model.layers[2].get_output(train=False))
# get_3rd_layer_output = theano.function([model.layers[0].input],
#                                      model.layers[3].get_output(train=False))
# get_4th_layer_output = theano.function([model.layers[0].input],
#                                     model.layers[4].get_output(train=False))
# get_5th_layer_output = theano.function([model.layers[0].input],
#                                     model.layers[5].get_output(train=False))
#
# o = get_1st_layer_output(X)
# p = get_2nd_layer_output(X)
# q = get_3rd_layer_output(X)
# r = get_4th_layer_output(cccc)
# s = get_5th_layer_output(cccc)
#
# # cccc = array(pickle.load(open(globals.store_path + globals.features_file)))
# # labels = array(pickle.load(open(globals.store_path + globals.labels_file)))
# #
# model = Sequential()
# model.add(Convolution2D(globals.dimension, 1, 2, globals.dimension))
# model.add(Activation('tanh'))
# model.add(AveragePooling2D(poolsize=(globals.s_size,1)))
# model.add(Flatten())
# model.add(Dense(2*globals.dimension,1))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
# model.fit(cccc, list(itertools.chain(labels)), nb_epoch=150, batch_size=32, show_accuracy=True)