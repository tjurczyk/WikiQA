from keras.layers.convolutional import *
from keras.layers.core import *
from keras.models import Sequential

import extras
import globals

def get_cnn(nn_type="regular"):
    model = None

    if nn_type == "regular":
        model = build_regular_model()
    elif nn_type == "multi-ngram":
        model = build_ngram_model()
    elif nn_type == "with-logistic":
        model = build_regular_model_with_logistic()
    else:
        raise ValueError("Unsupported model option. Provided: %s" % nn_type)

    globals.logger.info("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode="binary")
    globals.logger.info("Model compiled, start training...")
    return model


def build_regular_model():
    model = Sequential()

    # Convolution layer with its activation
    model.add(Convolution2D(globals.nb_filters, 1, 2, globals.dimension))
    model.add(Activation('tanh'))

    # Pooling
    model.add(extras.AveragePooling2D(poolsize=(globals.s_size, 1)))

    # Flattening and Dense Layer
    model.add(Flatten())
    model.add(Dense(2*globals.nb_filters, 1))

    # Activation at the end is sigmoid.
    model.add(Activation('sigmoid'))

    return model


def build_ngram_model():
    # TODO: ngram_filters must be a parameter
    ngram_filters = [2, 3, 4, 5]
    conv_filters = []

    # Create n multigram layers
    for n_gram in ngram_filters:
        conv_filters.append(Sequential())
        conv_filters[-1].add(Convolution2D(globals.dimension, 1, n_gram, globals.dimension))
        conv_filters[-1].add(Activation('tanh'))
        conv_filters[-1].add(extras.AveragePooling2D(poolsize=((globals.s_size + 1)-n_gram, 1)))
        conv_filters[-1].add(Flatten())
        conv_filters[-1].add(Dense(2*globals.dimension, 1))


    # Merge all of them into one and perform Dense
    model = Sequential()
    model.add(Merge(conv_filters, mode='concat'))

    model.add(Dense(len(ngram_filters), 1))
    model.add(Activation('sigmoid'))

    return model


def build_regular_model_with_logistic():
    # Part of network with regular NN
    nn_layer = Sequential()
    nn_layer.add(Convolution2D(globals.nb_filters, 1, 2, globals.dimension))
    nn_layer.add(Activation('tanh'))
    nn_layer.add(extras.AveragePooling2D(poolsize=(globals.s_size, 1)))
    nn_layer.add(Flatten())
    nn_layer.add(Dense(2*globals.nb_filters, 1))

    # Part of network with logistic regression
    # TODO: this (6, 6) must be parameterized (it's number of features in LR itself
    lr_layer = Sequential()
    lr_layer.add(Dense(6, 6))

    # Actual model merges two previous parts
    model = Sequential()
    model.add(Merge([nn_layer, lr_layer], mode='concat'))
    model.add(Dense(7, 1))
    model.add(Activation('sigmoid'))

    return model