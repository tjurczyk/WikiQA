from Base import Base
from Base_Sep import Base_Sep
from Attention import Attention
from misc import default_word, default_vocab, get_pbar, save_plots, log, read_model_data, write_model_data, read_lm_data, write_lm_data

def main(job_id, params):
    print(job_id)
    print(params)
    lstm = Attention(str(job_id))
    lstm.load_hyper(params)
    lstm.make_imports()

    return lstm.run()