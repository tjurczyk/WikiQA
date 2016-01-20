import logging
import os
from optparse import OptionParser
from enum import Enum

logging.basicConfig()
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Options parse
parser = OptionParser()
parser.add_option("-g", "--generate", help="generate mode (vectors/indexes)", dest="gen_mode")
parser.add_option("-l", "--qlimit", help="limit of number of questions (generate mode)", dest="q_limit")
parser.add_option("-e", "--experiment", help="experiment mode", dest="exp_mode")
parser.add_option("--train", help="train data location (experiment mode)", dest="train_data")
parser.add_option("--validate", help="validate data location (experiment mode)", dest="validate_data")
parser.add_option("--test", help="test data location (experiment mode)", dest="test_data")
parser.add_option("--nb_epoch", help="number of epochs (experiment mode)", dest="nb_epoch")
parser.add_option("--batch_size", help="batch size (experiment mode)", dest="batch_size")
options, arguments = parser.parse_args()

# Word embeddings size (used in generation and experiment)
dimension = 300

# Size of the sentence (either question or sentence).
# If the actual sentence has less than it, it will be padded with zeroes
s_size = 40

# Experiment settings
# CNN settings
nb_filters = 50
nb_epoch = 20
batch_size = 20
pooling_mode = 'average_exc_pad'

# LR settings
lr_features = [
    "wo",
    "wo_idf",
    "q_len",
    "dependency",
]

lr_dep_features = [
    #"string_word",
    "emb_word"
]

# Framework settings
# Paths anf files
data_path = 'data/'
models_path = 'models/'

nn_features_file = "features"
nn_labels_file = "labels"
lr_features_file = "lr_features"
lr_labels_file = "lr_labels"

input_files = {'train': 'WikiQA_data/WikiQASent-train.txt',
               'test': 'WikiQA_data/WikiQASent-test.txt',
               'validate': 'WikiQA_data/WikiQASent-dev.txt',
               'all': 'WikiQA_data/WikiQA.tsv'}

dep_input_files = {'train': 'WikiQA_data/train.cnlp',
                   'test': 'WikiQA_data/test.cnlp',
                   'validate': 'WikiQA_data/validate.cnlp'}

# Syntactic module settings
# comparator supported opts: "string_word", "emb_word"
#d_comparator = "string_word"
# metrics supported opts: "avg", "sum", "max"
d_metrics = "max"
# using word_form or lemma
d_typing = "word_form"

# Probably obsolete, will be removed
normalize_numbers = False

# Parse parameters
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

# Load stopwords
path_to_stopwords = os.path.dirname(os.path.abspath(__file__))
stop_words = set()
for l in open(path_to_stopwords + '/short-stopwords.txt').readlines():
    stop_words.add(l.strip())

from string import punctuation
p_marks = set(punctuation)


def get_config():
    config = {"mode": mode,
              "dimension": dimension,
              "s_size": s_size,
              "pooling_mode": pooling_mode}
    return config


def get_printy_dict(config_dict, ordering_list):
    output = ""
    print ("Config dict: %s" % config_dict)
    for order_item in ordering_list:
        output += order_item.ljust(18, ".")
        output += " " + str(config_dict[order_item]) + "\n"
    return output

config_dict = get_config()
print("Main settings in globals:")
p_order = ['mode', 'dimension', 's_size', 'pooling_mode']
print(get_printy_dict(config_dict, p_order))