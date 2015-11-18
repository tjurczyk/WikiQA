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

import globals
import extras
from models import cnn

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

    X_train = []
    X_validate = []
    X_test = []

    # Connect samples (to logistic regression) with predictions (output of NN)
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


def test_lr_on_data(X_train, y_train, X_validate, y_validate, X_test, y_test):
    y_train_flatten = list(itertools.chain(*y_train))

    # Train LR Model
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, y_train_flatten)

    # Test model on validation set
    predictions_val = lr.predict_proba(X_validate)
    predictions_val = array([i[-1] for i in predictions_val])
    best_threshold_validate = find_threshold_logistic(y_validate, predictions_val, predictions_val)
    precision_val, recall_val, f1_val = evaluate_with_threshold(y_validate, predictions_val, predictions_val,
                                                                best_threshold_validate)
    globals.logger.info("Found threshold: %f. Precision/recall/f1 over validation set: %f/%f/%f" %
                        (best_threshold_validate, precision_val, recall_val, f1_val))

    # Test model on test set
    predictions_test = lr.predict_proba(X_test)
    predictions_test = array([i[-1] for i in predictions_test])
    best_threshold_test = find_threshold_logistic(y_test, predictions_test, predictions_test, verbose=True)
    precision, recall, f1 = evaluate_with_threshold(y_test, predictions_test, predictions_test, best_threshold_test)
    globals.logger.info("Found threshold: %f. Precision/recall/f1 over test set: %f/%f/%f" %
                        (best_threshold_test, precision, recall, f1))

    return precision, recall, f1


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
    model = cnn.get_cnn("regular")

    y_train_flatted = list(itertools.chain(*y_train))
    nb_batch = len(X_train)/batch_size + 1

    best_f1 = 0.0
    best_f1_index = 0

    # Load questions and build idf for them
    questions_train, v, idf = load_questions_from_file("train", q_limit['train'])
    questions_validate, v, idf = load_questions_from_file("validate", q_limit['validate'])
    questions_test, v, idf = load_questions_from_file("test", q_limit['test'])

    build_idf([questions_train, questions_validate, questions_test])

    # Create samples for loaded questions
    samples_train = decompose_questions(questions_train)
    samples_validate = decompose_questions(questions_validate)
    samples_test = decompose_questions(questions_test)

    print("len of X_train: %d and samples_train: %d" % (len(X_train), len(samples_train)))

    for e in range(nb_epoch):
        print("Epoch %d" % e)
        progress_bar = generic_utils.Progbar(X_train.shape[0])

        # For training NN, shuffle the data
        X_train_shuffled, y_train_shuffled = shuffle_set(X_train, y_train_flatted)

        # Train for number of batches
        for i in range(nb_batch):
            train_loss,train_accuracy = model.train_on_batch(X_train_shuffled[i*batch_size:(i+1)*batch_size],
                                                             y_train_shuffled[i*batch_size:(i+1)*batch_size],
                                                             accuracy=True)
            progress_bar.add(batch_size, values=[("train loss", train_loss),("train accuracy:", train_accuracy)])

        # Check the score on the validation data
        #results_val = test_model(model, X_validate, y_validate)
        #best_threshold = find_threshold(y_validate, results_val["y_predicted_scores"], results_val["y_predicted_scores"])
        #precision_val, recall_val, f1_val = evaluate_with_threshold(y_validate, results_val["y_predicted_scores"],
        #                                                            results_val["y_predicted_scores"],
        #                                                            best_threshold)

        # Check the score on the test data
        #results_test = test_model(model, X_test, y_test)
        #precision_test, recall_test, f1_test = evaluate_with_threshold(y_test, results_test["y_predicted_scores"],
        #                                                               results_test["y_predicted_scores"],
        #                                                               best_threshold)

        #nn_string = "NN tests:\n" + "Threshold".ljust(40, ".") + " %.4f" + "\nOver validation set\n" \
        #            + "validation loss, validation acc".ljust(40, ".") + " %.4f %.4f\n" \
        #            + "precision, recall, f1".ljust(40, ".") + " %.4f %.4f %.4f\n" \
        #            + "Over test set\n" \
        #            + "test loss, test acc".ljust(40, ".") + " %.4f %.4f\n" \
        #            + "precision, recall, f1".ljust(40, ".") + " %.4f %.4f %.4f\n" \

        #globals.logger.info(nn_string % (best_threshold, results_val['test_loss'], results_val['test_accuracy'],
        #                                 precision_val, recall_val, f1_val,
        #                                 results_test['test_loss'], results_test['test_accuracy'],
        #                                 precision_test, recall_test, f1_test))

        # Now evaluate with logistic regression
        # Get predictions from NN
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


def shuffle_set(X, y):
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


def find_threshold_logistic(questions_gold_sets, predictions, confidence, verbose=False):
    thr = 0.05
    best_f1 = -1
    best_f1_thr = -1
    while thr < 0.16:
        precision, recall, f1 = evaluate_with_threshold(questions_gold_sets, predictions, confidence, thr)
        if verbose is True:
            print("For thre: %f, prec/rec/f1: %f/%f/%f" % (thr, precision, recall, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = thr

        thr += 0.01

    print("After search best threshold: %f with f1: %f" % (best_f1_thr, best_f1))

    return best_f1_thr


def find_threshold(questions_gold_sets, predictions, confidence):
    index_begin = 0
    sum_of_scores = 0.0
    all_questions_number = 0

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
    avg_precision, avg_recall, avg_f1 = evaluate_with_threshold(
        questions_gold_sets, predictions, confidence, average_threshold)

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
    all_questions_with_answers = 0.0
    predicted_questions = 0.0
    correctly_predicted_questions = 0.0

    for question_set in questions_gold_sets:
        # Get the slice from predictions and confidence for this question_set
        index_end = index_begin + len(question_set)
        predictions_slice = predictions[index_begin:index_end].flatten()
        confidence_slice = confidence[index_begin:index_end].flatten()

        # Find the maximum value prediction
        max_val_index = numpy.argmax(predictions_slice)
        #if len(numpy.argwhere(predictions_slice == predictions_slice[max_val_index])) > 1:
        #    raise ValueError("More than one max values in slice!\nslice: %s" % predictions_slice)

        if confidence_slice[max_val_index] > threshold:
            predicted_answer = max_val_index
        else:
            predicted_answer = -1

        # Check if this question has an answer.
        gold_answer_ids = [x for x, y in enumerate(question_set) if y == 1]
        nb_gold_correct_answers = len(gold_answer_ids)

        # If there is an answer, increment number of all questions
        if nb_gold_correct_answers > 0:# and nb_gold_correct_answers != len(question_set):
            all_questions_with_answers += 1

        # If the question predicted with the answer, increment predicted_questions
        if predicted_answer > -1:
            predicted_questions += 1

        # If the question predicted correctly, increment correctly_predicted_questions
        if predicted_answer > -1 and predicted_answer in gold_answer_ids:
            correctly_predicted_questions += 1

        index_begin = index_end

    # Calculate precision, recall and F1
    if correctly_predicted_questions == 0 or predicted_questions == 0:
        return 0.0, 0.0, 0.0
    precision = float(correctly_predicted_questions)/predicted_questions
    recall = float(correctly_predicted_questions)/all_questions_with_answers
    f1 = (2*precision*recall)/(precision+recall) if (precision > 0.0 and recall > 0.0) else 0.0

    return precision, recall, f1


def test_nn_logistic():
    # Load data for NN
    X_train, y_train = load_data(train_data, globals.nn_features_file, globals.nn_labels_file)
    X_validate, y_validate = load_data(validate_data, globals.nn_features_file, globals.nn_labels_file)
    X_test, y_test = load_data(test_data, globals.nn_features_file, globals.nn_labels_file)

    train_and_test(X_train, y_train, X_validate, y_validate, X_test, y_test)


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

    if globals.exp_mode == "test_nn_logistic":
        test_nn_logistic()
    elif globals.exp_mode == "test_idf":
        test_idf_build()
    else:
        raise RuntimeError("Unknown exp_mode option.")
