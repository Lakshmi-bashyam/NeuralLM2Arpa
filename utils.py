import json
import os

from numpy import average
from torchtext.data import get_tokenizer
from speechbrain.lm.ngram import BackoffNgramLM
from speechbrain.lm.arpa import read_arpa


def backoff_model(path):
    # tokenizer = get_tokenizer("basic_english")
    # with open(os.path.join(root_dir, 'final_corpus.txt')) as f:
    #     data = f.read().split('\n')

    # Create LM
    with open(path) as f:
        num_grams, ngrams, backoffs = read_arpa(f)
    lm = BackoffNgramLM(ngrams, backoffs)
    return lm

def replace_lm_prob(path, lm):
    with open(os.path.join(path, 'ngram_prob.json')) as f:
        ngram_prob = json.load(f)
    ngram_prob_avg = {}
    for k, v in ngram_prob.items():
        k = tuple(k.split(" "))
        k = (k[-1], [k[:-1]])
        ngram_prob_avg[k] = average(v)
    with open(os.path.join(path, 'ngram_prob_avg.json'), 'w+', encoding="utf8") as outfile:
        json.dump(ngram_prob_avg, outfile)
    

if __name__ == '__main__':
    lm = backoff_model('data')
    ngram_prob = replace_lm_prob('data', lm)
    print(ngram_prob)

    