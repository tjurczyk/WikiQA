from keras.layers.convolutional import *
from keras.layers.core import *
from keras.models import Sequential
from numpy import array, ndarray, argmax
import numpy
import itertools
import cPickle as pickle
from pprint import pprint
from keras.utils import generic_utils
from generate_input_file import load_questions_from_file
from decomposition.sentence import decompose_questions, build_idf
from sklearn.linear_model import LogisticRegression
from random import randint
from random import shuffle
from sklearn import svm
import sys

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
file_prefix = str(randint(1, 1000000))


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
    model = train(X_train, y_train, X_validate, y_validate, validation_mode=globals.validation_mode)

    # Test on the validation set
    globals.logger.info("Testing NN model on validation set...")
    results = test_model(model, X_validate, y_validate)
    globals.logger.info("Model tested on validation set.")

    # Try to find a threshold on the validation set
    globals.logger.info("Finding best threshold on the validation set...")
    best_threshold = find_threshold(y_validate, results["y_predicted_scores"], results["y_predicted_scores"])

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
    samples = decompose_questions(questions_train)
    dict_nn = test_model(model, X_train, y_train)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(train_data, {'features': samples, 'labels': y_train}, globals.lr_features_file, globals.lr_labels_file)

    # Validation set generation
    globals.logger.info("Generating the validation set for Logistic Regression")
    samples = decompose_questions(questions_validate)

    dict_nn = test_model(model, X_validate, y_validate)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(validate_data, {'features': samples, 'labels': y_validate},
              globals.lr_features_file, globals.lr_labels_file)

    # Test section
    globals.logger.info("Generating the test set for Logistic Regression")
    samples = decompose_questions(questions_test)

    dict_nn = test_model(model, X_test, y_test)
    nn_scores = dict_nn['y_predicted_scores'].flatten()

    for idx, sample in enumerate(samples):
        sample.append(nn_scores[idx])

    dump_data(test_data, {'features': samples, 'labels': y_test}, globals.lr_features_file, globals.lr_labels_file)


def validate_on_lr(samples_train, samples_validate, samples_test,
                   predictions_train, predictions_validate, predictions_test,
                   y_train, y_validate, y_test):
    """
    Validate output of NN using Logistic Regression.
    :return: f1 of the score trained on the output from NN (train), tested on validation set.
    """

    if len(samples_train) != len(predictions_train) or len(samples_validate) != len(predictions_validate):
        raise ValueError("Validate on LR got wrong argument lengths. %d != %d (train) or %d != %d (validate)" %
                         (len(samples_train), len(predictions_train), len(samples_validate), len(predictions_validate)))


    y_train_flatten = list(itertools.chain(*y_train))

    X_train = []
    X_validate = []
    X_test = []

    # Connect samples and predictions
    for sample, prediction in zip(samples_train, predictions_train):
        train_sample = sample[:]
        train_sample.append(prediction[0])
        X_train.append(train_sample)

    for sample, prediction in zip(samples_validate, predictions_validate):
        validate_sample = sample[:]
        validate_sample.append(prediction[0])
        X_validate.append(validate_sample)

    for sample, prediction in zip(samples_test, predictions_test):
        test_sample = sample[:]
        test_sample.append(prediction[0])
        X_test.append(test_sample)

    globals.logger.info("Training and testing Logistic Regression to validate...")
    precision, recall, f1 = test_lr_on_data(X_train, y_train, X_validate, y_validate, X_test, y_test)
    return precision, recall, f1


def test_lr():
    # Load the data
    X_train, y_train = load_data(train_data, globals.lr_features_file, globals.lr_labels_file)
    X_validate, y_validate = load_data(validate_data, globals.lr_features_file, globals.lr_labels_file)
    X_test, y_test = load_data(test_data, globals.lr_features_file, globals.lr_labels_file)

    precision, recall, f1 = test_lr_on_data(X_train, y_train, X_validate, y_validate, X_test, y_test)
    return precision, recall, f1


def test_lr_on_data(X_train, y_train, X_validate, y_validate, X_test, y_test):
    y_train_flatten = list(itertools.chain(*y_train))

    lr = LogisticRegression(solver='lbfgs')#, class_weight={1: 0.9})#)#solver='lbfgs', class_weight={1: 0.5})
    lr.fit(X_train, y_train_flatten)

    ######################################################
    # This is only for DEBUG purposes
    # Try to see if finding threshold on Test will improve
    predictions_test = lr.predict_proba(X_test)
    predictions_test = array([i[-1] for i in predictions_test])
    #confidence_test = lr.decision_function(X_test)
    best_threshold_test = new_find_threshold(y_test, predictions_test, predictions_test)
    precision_test, recall_test, f1_test = evaluate_with_threshold(y_test, predictions_test, predictions_test, best_threshold_test)
    globals.logger.info("DEBUG Found threshold: %f. Precision/recall/f1 on found threshold "
                        "over test set: %f/%f/%f" % (best_threshold_test, precision_test, recall_test, f1_test))
    ######################################################

    # This is actual part: find threshold on validate and test on test
    predictions_val = lr.predict_proba(X_validate)
    predictions_val = array([i[-1] for i in predictions_val])
    #confidence_val = lr.decision_function(X_validate)
    best_threshold = new_find_threshold(y_validate, predictions_val, predictions_val)

    precision, recall, f1 = evaluate_with_threshold(y_validate, predictions_val, predictions_val, best_threshold)
    globals.logger.info("Precision/recall/f1 over validation set: %f/%f/%f" % (precision, recall, f1))

    predictions_test = lr.predict_proba(X_test)
    predictions_test = array([i[-1] for i in predictions_test])
    #confidence_test = lr.decision_function(X_test)
    #print_logistic_predictions(predictions_test, y_test[:30])
    precision, recall, f1 = evaluate_with_threshold(y_test, predictions_test, predictions_test, best_threshold)

    return precision, recall, f1


def test_idf():
    questions_train = load_questions_from_file("train", q_limit['train'])
    questions_validate = load_questions_from_file("validate", q_limit['validate'])
    questions_test = load_questions_from_file("test", q_limit['test'])

    X_train, y_train = load_data(train_data, globals.nn_features_file, globals.nn_labels_file)
    X_validate, y_validate = load_data(validate_data, globals.nn_features_file, globals.nn_labels_file)
    X_test, y_test = load_data(test_data, globals.nn_features_file, globals.nn_labels_file)

    y_train_flatten = list(itertools.chain(*y_train))

    samples_train = decompose_questions(questions_train)
    samples_validate = decompose_questions(questions_validate)
    samples_test = decompose_questions(questions_test)

    lr = LogisticRegression()
    lr.fit(samples_train, y_train_flatten)

    predictions = lr.decision_function(samples_validate)
    best_threshold = find_threshold(y_validate, predictions)
    precision, recall, f1 = evaluate_with_threshold(y_validate, predictions, best_threshold)
    globals.logger.info("Found threshold: %f. Precision/recall/f1 on found threshold "
                        "over validation set: %f/%f/%f" % (best_threshold, precision, recall, f1))

    predictions = lr.decision_function(samples_test)
    precision, recall, f1 = evaluate_with_threshold(y_test, predictions, best_threshold)
    globals.logger.info("Precision/recall/f1 on found threshold "
                        "over test set: %f/%f/%f" % (precision, recall, f1))


def test_model(model, X_test, y_test):
    # Get test loss and test accuracy
    (test_loss, test_accuracy) = model.evaluate(X_test, list(itertools.chain(*y_test)),
                                                show_accuracy=True, batch_size=batch_size)

    # Get predicted classes and scores
    y_predicted_scores = model.predict(X_test, batch_size=batch_size)

    return {"test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "y_predicted_scores": y_predicted_scores}


def train_and_test(X_train, y_train, X_validate, y_validate, X_test, y_test):
    #model = get_ngram_model()
    model = get_regular_model()
    #model = get_regular_model_with_logistic()

    y_train_flatted = list(itertools.chain(*y_train))
    y_validate_flatted = list(itertools.chain(*y_validate))
    nb_batch = len(X_train)/batch_size + 1

    best_loss = 1.0
    best_loss_index = -1
    best_f1 = 0.0
    best_f1_index = 0
    previous_loss = 0

    questions_train, v, idf = load_questions_from_file("train", q_limit['train'])
    questions_validate, v, idf = load_questions_from_file("validate", q_limit['validate'])
    questions_test, v, idf = load_questions_from_file("test", q_limit['test'])

    build_idf([questions_train, questions_validate, questions_test])

    samples_train = decompose_questions(questions_train)
    samples_validate = decompose_questions(questions_validate)
    samples_test = decompose_questions(questions_test)

    print("len of X_train: %d and samples_train: %d" % (len(X_train), len(samples_train)))

    # Section for Logistic integrated in NN
    #samples_train_shaped = array(samples_train).reshape(len(samples_train), 1, 1, 6)
    #samples_validate_shaped = array(samples_validate).reshape(len(samples_validate), 1, 1, 6)
    #samples_test_shaped = array(samples_test).reshape(len(samples_test), 1, 1, 6)

    for e in range(nb_epoch):
        print("Epoch %d" % e)
        progress_bar = generic_utils.Progbar(X_train.shape[0])

        # For training NN, shuffle the data
        X_train_shuffled, y_train_shuffled = shuffleSet(X_train, y_train_flatted)
        #X_train_shuffled, y_train_shuffled = shuffle_and_sample(X_train, y_train_flatted, 10001)

        #nb_batch = len(X_train_shuffled)/batch_size + 1

        #progress_bar = generic_utils.Progbar(X_train_shuffled.shape[0])

        for i in range(nb_batch):
            train_loss,train_accuracy = model.train_on_batch(X_train_shuffled[i*batch_size:(i+1)*batch_size],
                                                             y_train_shuffled[i*batch_size:(i+1)*batch_size],
                                                             accuracy=True)
            progress_bar.add(batch_size, values=[("train loss", train_loss),("train accuracy:", train_accuracy)])

        # Check the score on the validation data
        results_val = test_model(model, X_validate, y_validate)
        best_threshold = find_threshold(y_validate, results_val["y_predicted_scores"], results_val["y_predicted_scores"])
        precision_val, recall_val, f1_val = evaluate_with_threshold(y_validate, results_val["y_predicted_scores"],
                                                                    results_val["y_predicted_scores"],
                                                                    best_threshold)

        # Check the score on the test data
        results_test = test_model(model, X_test, y_test)
        precision_test, recall_test, f1_test = evaluate_with_threshold(y_test, results_test["y_predicted_scores"],
                                                                       results_test["y_predicted_scores"],
                                                                       best_threshold)

        nn_string = "NN tests:\n" + "Threshold".ljust(40, ".") + " %.4f" + "\nOver validation set\n" \
                    + "validation loss, validation acc".ljust(40, ".") + " %.4f %.4f\n" \
                    + "precision, recall, f1".ljust(40, ".") + " %.4f %.4f %.4f\n" \
                    + "Over test set\n" \
                    + "test loss, test acc".ljust(40, ".") + " %.4f %.4f\n" \
                    + "precision, recall, f1".ljust(40, ".") + " %.4f %.4f %.4f\n" \

        globals.logger.info(nn_string % (best_threshold, results_val['test_loss'], results_val['test_accuracy'],
                                         precision_val, recall_val, f1_val,
                                         results_test['test_loss'], results_test['test_accuracy'],
                                         precision_test, recall_test, f1_test))

        # Now try with logistic regression
        predictions_train = model.predict(X_train)
        predictions_validate = model.predict(X_validate)
        predictions_test = model.predict(X_test)

        # Evaluate on logistic regression
        precision, recall, f1 = validate_on_lr(samples_train, samples_validate, samples_test,
                            predictions_train, predictions_validate, predictions_test,
                            y_train, y_validate, y_test)

        lr_string = "LR tests:\n" + "Over test set\n" \
                    + "precision, recall, f1".ljust(40, ".") + " %.4f %.4f %.4f\n"

        globals.logger.info(lr_string % (precision, recall, f1))

        if f1 >= best_f1:
            best_f1 = f1
            best_f1_index = e

    globals.logger.info("Training done, best f1 on logistic regression is: %.4f for epoch nr: %d" %
                   (best_f1, best_f1_index))


def shuffle_and_sample(X, y, n=10000):
    length = len(X)

    X_sampled, y_sampled = [], []

    y_ones_indexes = [idx for idx, x in enumerate(y) if x == 1]
    y_ones_indexes_set = set(y_ones_indexes)
    X_ones = [i for idx, i in enumerate(X) if idx in y_ones_indexes]

    # random n negative samples

    sampled = 0
    while True:
        randomized = randint(0, length-1)
        if randomized not in y_ones_indexes_set:
            X_sampled.append(X[randomized])
            y_sampled.append(0)
            sampled += 1

        if sampled == n - len(y_ones_indexes):
            break

    # Add true (1) samples
    X_sampled.extend(X_ones)
    y_sampled.extend([1] * len(X_ones))

    # And shuffle
    X_shuffled, y_shuffled = shuffleSet(X_sampled, y_sampled)

    return X_shuffled, y_shuffled


def shuffleSet(X, y):
    if len(X) != len(y):
        raise ValueError("Lengths of X and y don't match")

    X_shuffled = []
    y_shuffled = []
    index_shuffled = range(len(X))
    shuffle(index_shuffled)
    for i in index_shuffled:
        X_shuffled.append(X[i])
        y_shuffled.append(y[i])

    return array(X_shuffled), array(y_shuffled)


def get_regular_model():
    model = Sequential()
    model.add(Convolution2D(globals.nb_filters, 1, 2, globals.dimension))
    model.add(Activation('tanh'))

    model.add(extras.AveragePooling2D(poolsize=(globals.s_size, 1)))

    model.add(Flatten())

    model.add(Dense(2*globals.nb_filters, 1))

    model.add(Activation('sigmoid'))

    globals.logger.info("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
    globals.logger.info("Model compiled, start training...")
    return model


def get_regular_model_with_logistic():
    nn_layer = Sequential()
    nn_layer.add(Convolution2D(globals.nb_filters, 1, 2, globals.dimension))
    nn_layer.add(Activation('tanh'))

    nn_layer.add(extras.AveragePooling2D(poolsize=(globals.s_size, 1)))

    nn_layer.add(Flatten())

    nn_layer.add(Dense(2*globals.nb_filters, 1))

    #nn_layer.add(Activation('sigmoid'))

    lr_layer = Sequential()
    lr_layer.add(Dense(6, 6))
    #lr_layer.add(Activation('linear'))

    model = Sequential()
    model.add(Merge([nn_layer, lr_layer], mode='concat'))
    model.add(Dense(7, 1))
    model.add(Activation('sigmoid'))


    globals.logger.info("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
    globals.logger.info("Model compiled, start training...")
    return model


def get_ngram_model():
    ngram_filters = [2, 3, 4, 5]
    conv_filters = []

    for n_gram in ngram_filters:
        conv_filters.append(Sequential())
        conv_filters[-1].add(Convolution2D(globals.dimension, 1, n_gram, globals.dimension))
        conv_filters[-1].add(Activation('tanh'))
        conv_filters[-1].add(extras.AveragePooling2D(poolsize=((globals.s_size + 1)-n_gram, 1)))
        conv_filters[-1].add(Flatten())
        conv_filters[-1].add(Dense(2*globals.dimension, 1))


    model = Sequential()
    model.add(Merge(conv_filters, mode='concat'))

    model.add(Dense(len(ngram_filters), 1))



    # model = Sequential()
    # model.add(Convolution2D(globals.dimension, 1, 4, globals.dimension))
    # model.add(Activation('tanh'))

    #model.add(extras.AveragePooling2DVariable(poolsize=(39, 1), stride=(38, 1)))
    #model.add(extras.AveragePooling2D(poolsize=(38, 1)))#, stride=(39,1)))

    #model.add(Flatten())

    #model.add(Dense(2*globals.dimension, 1))




    model.add(Activation('sigmoid'))

    globals.logger.info("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
    globals.logger.info("Model compiled, start training...")
    return model


def train(X_train, y_train, X_validate, y_validate, validation_mode):
    model = Sequential()
    model.add(Convolution2D(globals.dimension, 1, 2, globals.dimension))
    model.add(Activation('tanh'))

    model.add(extras.AveragePooling2D(poolsize=(globals.s_size,1)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(2*globals.dimension, 1))

    model.add(Activation('sigmoid'))

    model.add(Dropout(0.5))

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

    if validation_mode == "validation_data":
        for e in range(nb_epoch):
            print("Epoch %d" % e)
            progress_bar = generic_utils.Progbar(X_train.shape[0])
            for i in range(nb_batch):
                train_loss,train_accuracy = model.train_on_batch(X_train[i*batch_size:(i+1)*batch_size],
                                                                 y_train_flatted[i*batch_size:(i+1)*batch_size],
                                                                 accuracy=True)
                progress_bar.add(batch_size, values=[("train loss", train_loss),("train accuracy:", train_accuracy)])

            if validation_mode == "validation_data":
                # Check the scores after this batch
                results = test_model(model, X_validate, y_validate)
                best_threshold = find_threshold(y_validate, results["y_predicted_scores"])
                precision, recall, f1 = evaluate_with_threshold(y_validate, results["y_predicted_scores"], best_threshold)
                globals.logger.info("OVER Validation found threshold: %.4f.\n"
                                    "Validation loss: %.4f - Validation accuracy: %.4f\n"
                                    "Precision: %.4f - Recall: %.4f - F1: %.4f" % (best_threshold,
                                                                                   results['test_loss'],
                                                                                   results['test_accuracy'],
                                                                                   precision, recall, f1))

                if e == 1:
                    previous_loss = results['test_loss']
                if f1 >= best_f1 \
                        and e > 1 \
                        and float(results['test_loss']) < float(previous_loss) \
                        and float(previous_loss)-float(results['test_loss']) <= globals.learning_margin:
                    # It is better score that is qualified
                    # <= 0.0003, because we don't want to select any accidental early-stage high score.
                    best_f1 = f1
                    best_f1_index = e
                    model.save_weights("model_weights-" + file_prefix + ".h5", overwrite=True)

                previous_loss = results['test_loss']

        globals.logger.info("Training done, best f1 on validation is: %.4f for epoch nr: %d" %
                       (best_f1, best_f1_index))
        model.load_weights("model_weights-" + file_prefix + ".h5")

    elif validation_mode == "cross_validation":
        history = model.fit(X_train, y_train_flatted, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.3)
        print("History: ")
        pprint(history.history)
        globals.logger.info("Training done.")
    else:
        raise RuntimeError("Unknown validation_mode in train(), given: %s" % validation_mode)

    return model


def new_find_threshold(questions_gold_sets, predictions, confidence):

    thr = 0.02
    best_f1 = -1
    best_f1_thr = -1
    while thr < 0.40:
        precision, recall, f1 = evaluate_with_threshold(questions_gold_sets, predictions, confidence, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = thr

        thr += 0.01

    print("After search best threshold: %f with f1: %f" % (best_f1_thr, best_f1))

    return best_f1_thr


def find_threshold(questions_gold_sets, predictions, confidence):
    #print("Find threshold, received predictions: ")
    #pprint(predictions)
    #print("confidence")
    #pprint(confidence)

    predictions_flatten = predictions.flatten()
    confidence_flatten = confidence.flatten()

    # Make a pass through all corrects and calculate their average
    index_begin = 0
    sum_of_scores = 0.0
    all_questions_number = 0

    #print("Type is: %s" % type(predictions[0]))
    #print("Received predictions: %s" % predictions)

    #if type(predictions[0]) is ndarray:
    #    predictions = array([i[-1] for i in predictions])

    #print("After checking: %s" % predictions)

    # Sum all confidence scores of correct answers
    for question_set in questions_gold_sets:
        index_end = index_begin + len(question_set)
        if 1 in question_set:
            gold_answer_id = question_set.index(1)
        else:
            gold_answer_id = -1

        confidence_slice = confidence[index_begin:index_end]

        if gold_answer_id > -1:
            sum_of_scores += confidence_slice[gold_answer_id]
            all_questions_number += 1

        index_begin = index_end

    average_threshold = sum_of_scores/all_questions_number
    #globals.logger.info("Found initial average threshold of: %f. Now searching best threshold..." % average_threshold)
    avg_precision, avg_recall, avg_f1 = evaluate_with_threshold(questions_gold_sets, predictions, confidence, average_threshold)

    # If average f1 is 0.0, there is no need to search for
    if avg_f1 == 0.0:
        return average_threshold

    delta = 0.01

    # Go incrementing threshold
    current_threshold = average_threshold
    up_f1 = avg_f1
    up_threshold = current_threshold
    for i in xrange(20):
        current_threshold += delta
        precision, recall, f1 = evaluate_with_threshold(questions_gold_sets, predictions, confidence, current_threshold)
        if f1 > up_f1:
            up_f1 = f1
            up_threshold = current_threshold
        elif f1 < up_f1:
            break

    # Go decrementing threshold
    current_threshold = average_threshold
    down_f1 = avg_f1
    down_threshold = current_threshold
    for i in xrange(20):
        current_threshold -= delta
        precision, recall, f1 = evaluate_with_threshold(questions_gold_sets, predictions, confidence, current_threshold)
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


def evaluate_with_threshold(questions_gold_sets, predictions, confidence, threshold):
    index_begin = 0
    all_questions_with_answers = 0
    predicted_questions = 0
    correctly_predicted_questions = 0

    for question_set in questions_gold_sets:
        # Get the slice from predictions and confidence for this question_set
        index_end = index_begin + len(question_set)
        predictions_slice = predictions[index_begin:index_end].flatten()
        confidence_slice = confidence[index_begin:index_end].flatten()

        # Find the maximum value prediction
        max_val_index = numpy.argmax(predictions_slice)
        if len(numpy.argwhere(predictions_slice == predictions_slice[max_val_index])) > 1:
            raise ValueError("More than one max values in slice!")

        if confidence_slice[max_val_index] > threshold:
            predicted_answer = max_val_index
        else:
            predicted_answer = -1

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
        if predicted_answer > -1:
            predicted_questions += 1

        # If the question predicted correctly, increment correctly_predicted_questions
        if predicted_answer > -1 and predicted_answer == gold_answer_id:
            correctly_predicted_questions += 1

        index_begin = index_end

    # Calculate precision, recall and F1
    if correctly_predicted_questions == 0 or predicted_questions == 0:
        return 0.0, 0.0, 0.0
    precision = float(correctly_predicted_questions)/predicted_questions
    recall = float(correctly_predicted_questions)/all_questions_with_answers
    f1 = (2*precision*recall)/(precision+recall) if (precision > 0.0 and recall > 0.0) else 0.0

    return precision, recall, f1


def normalize_predictions(predictions, y):
    pass


def test_nn_logistic():
    # Load data for NN
    X_train, y_train = load_data(train_data, globals.nn_features_file, globals.nn_labels_file)
    X_validate, y_validate = load_data(validate_data, globals.nn_features_file, globals.nn_labels_file)
    X_test, y_test = load_data(test_data, globals.nn_features_file, globals.nn_labels_file)

    train_and_test(X_train, y_train, X_validate, y_validate, X_test, y_test)


def print_logistic_predictions(y_predictions, y_gold):
    index_begin = 0

    for question_set in y_gold:

        print("Question:")

        for i in xrange(len(question_set)):
            print("%s %s" % (str(y_predictions[index_begin+i]).ljust(20, " "), question_set[i]))

        index_begin += len(question_set)


def test_idf_build():
    questions_train, vocabulary, idf = load_questions_from_file("train", q_limit['train'])
    questions_validate, vocabulary, idf = load_questions_from_file("validate", q_limit['validate'])
    questions_test, vocabulary, idf = load_questions_from_file("test", q_limit['test'])
    X_train, y_train = load_data(train_data, globals.nn_features_file, globals.nn_labels_file)
    #X_validate, y_validate = load_data(validate_data, globals.nn_features_file, globals.nn_labels_file)
    #X_test, y_test = load_data(test_data, globals.nn_features_file, globals.nn_labels_file)

    y_train_flatten = list(itertools.chain(*y_train))

    build_idf([questions_train, questions_validate, questions_test])
    samples = decompose_questions(questions_train)

    print("Samples len: %d, y_train_flatten len: %d" % (len(samples), len(y_train_flatten)))

    #samples = samples[:3]

    #pprint(samples[:40])

    e = 0

    print("%s" % "wo".ljust(10, " ") + " " + "wo_idf".ljust(30, " ") + " " + "q_len".ljust(10, " ")
          + " " + "wo_answers".ljust(30, " ") + " " + "idf_avg".ljust(30, " ")
          + " " + "idf_norm".ljust(30, " ") + " " + "label")
    for idx, sample in enumerate(samples):
        print("%s, %s" % (str(sample[0]).ljust(10, " ") + " " + str(sample[1]).ljust(30, " ")
                          + " " + str(sample[2]).ljust(10, " ") + " " + str(sample[3]).ljust(30, " ")
                          + " " + str(sample[4]).ljust(30, " ") + " " + str(sample[5]).ljust(30, " ")
                          + " ", y_train_flatten[e]))
        e += 1

    #pprint(samples[:30])


def get_config():
    config = {"train_data": train_data,
              "validation_data": validate_data,
              "test_data": test_data,
              "exp_mode": globals.exp_mode,
              "nb_epoch": nb_epoch,
              "nb_filters": globals.nb_filters,
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

    print("Experiment mode options:")
    p_order = ['exp_mode', 'nb_epoch', 'nb_filters', 'batch_size', 'train_data', 'validation_data', 'test_data']
    print(globals.get_printy_dict(get_config(), p_order))

    if globals.exp_mode == "test_nn":
        test_nn()
    elif globals.exp_mode == "test_nn_logistic":
        test_nn_logistic()
    elif globals.exp_mode == "generate_train_logistic":
        test_nn(generate_logistic_data=True)
    elif globals.exp_mode == "test_idf":
        test_idf_build()
    elif globals.exp_mode == "test_lr":
        test_lr()
    else:
        raise RuntimeError("Unknown exp_mode option.")
