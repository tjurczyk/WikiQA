import logging
from optparse import OptionParser

logging.basicConfig()
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Default options
data_path = 'data/'
models_path = 'models/'
nb_epoch = 20
batch_size = 20
learning_margin = 0.0003
validation_mode = "validation_data"

normalize_numbers = False

# NN settings
pooling_mode = 'average_exc_pad'

# Options parse
parser = OptionParser()

parser.add_option("-g", "--generate", help="generate mode (train/test/dev/all)", dest="gen_mode")
parser.add_option("-l", "--qlimit", help="limit of number of questions (generate mode)", dest="q_limit")
parser.add_option("-e", "--experiment", help="experiment mode", dest="exp_mode")
parser.add_option("--train", help="train data location (experiment mode)", dest="train_data")
parser.add_option("--validate", help="validate data location (experiment mode)", dest="validate_data")
parser.add_option("--test", help="test data location (experiment mode)", dest="test_data")
parser.add_option("--nb_epoch", help="number of epochs (experiment mode)", dest="nb_epoch")
parser.add_option("--batch_size", help="batch size (experiment mode)", dest="batch_size")
parser.add_option("--validation_mode", help="validation mode (experiment mode)", dest="validation_mode")

# Parse parameters
options, arguments = parser.parse_args()
mode = None

if options.gen_mode is not None:
    gen_mode = gen_mode = options.gen_mode
    mode = "gen"
else:
    gen_mode = None

if options.exp_mode is not None:
    exp_mode = options.exp_mode
    mode = "exp"
else:
    exp_mode = None

if options.validation_mode is not None:
    validation_mode = options.validation_mode

# File names
nn_features_file = "features"
nn_labels_file = "labels"
lr_features_file = "lr_features"
lr_labels_file = "lr_labels"

# Dimension of the word representation
dimension = 300
nb_filters = 50

# Size of the sentence (either question or sentence).
# If the actual sentence has less than 20, it will be padded with zeroes
s_size = 40

input_files = {'train': 'WikiQA_data/WikiQASent-train.txt',
               'test': 'WikiQA_data/WikiQASent-test.txt',
               'validate': 'WikiQA_data/WikiQASent-dev.txt',
               'all': 'WikiQA_data/WikiQA.tsv'}

# Load stopwords
stop_words = set()
for l in open('short-stopwords.txt').readlines():
    stop_words.add(l.strip())

#print("Stopwords: %s" % stop_words)

from string import punctuation
p_marks = set(punctuation)


def get_config():
    config = {"mode": mode,
              "dimension": dimension,
              "s_size": s_size,
              "pooling_mode": pooling_mode,
              "learning_margin": learning_margin,
              "validation_mode": validation_mode}
    return config


def get_printy_dict(config_dict, ordering_list):
    output = ""
    for order_item in ordering_list:
        output += order_item.ljust(18, ".")
        output += " " + str(config_dict[order_item]) + "\n"

    return output

config_dict = get_config()
print("Main settings in globals:")
p_order = ['mode', 'dimension', 's_size', 'pooling_mode', 'learning_margin', 'validation_mode']
print(get_printy_dict(config_dict, p_order))