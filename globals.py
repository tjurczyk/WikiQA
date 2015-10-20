import os.path
import logging
from optparse import OptionParser
from pprint import pprint

logging.basicConfig()
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

# Default options
data_path = 'data/'
models_path = 'models/'
nb_epoch = 50
batch_size = 32

# Options parse
parser = OptionParser()

parser.add_option("-g", "--generate", help="generate mode (train/test/dev/all)", dest="gen_mode")
parser.add_option("-l", "--qlimit", help="number of questions (limit)", dest="q_limit")
parser.add_option("-e", "--experiment", help="experiment mode", dest="exp_mode")
parser.add_option("--train", help="training data location", dest="train_data")
parser.add_option("--validate", help="validate data location", dest="validate_data")
parser.add_option("--test", help="training data location", dest="test_data")
parser.add_option("--nb_epoch", help="number of epochs", dest="nb_epoch")
parser.add_option("--batch_size", help="batch size", dest="batch_size")

# -g
# generate either of three data sets or all

# -e:
# test_nn   : Train and test nn
# train     : Train on training data and validate on validate (if provided) only
# train-full: + prepare train data for logistic regression
# full      : + prepare test data for logistic regression

# Parse parameters
options, arguments = parser.parse_args()
mode = None

if options.gen_mode is not None:
    gen_mode = gen_mode = options.gen_mode
    mode = "gen"
else:
    gen_mode = None

if not options.q_limit:
    q_limit = -1
else:
    q_limit = int(options.q_limit)

if options.exp_mode is not None:
    exp_mode = options.exp_mode
    mode = "exp"
else:
    exp_mode = None

# File names
nn_features_file = "features"
nn_labels_file = "labels"
lr_features_file = "lr_features"
lr_labels_file = "lr_labels"

# Dimension of the word representation
dimension = 300

# Size of the sentence (either question or sentence).
# If the actual sentence has less than 20, it will be padded with zeroes
s_size = 40

input_files = {'train': 'WikiQA_data/WikiQA-train.tsv',
               'test': 'WikiQA_data/WikiQA-test.tsv',
               'validate': 'WikiQA_data/WikiQA-dev.tsv',
               'all': 'WikiQA_data/WikiQA.tsv'}

from stop_words import get_stop_words
stop_words = set(get_stop_words('en'))

from string import punctuation
p_marks = set(punctuation)


def get_config():

    config = {"mode": mode,
              "q_limit": q_limit if q_limit > -1 else "all",
              "dimension": dimension,
              "s_size": s_size}
    return config

pprint(get_config())