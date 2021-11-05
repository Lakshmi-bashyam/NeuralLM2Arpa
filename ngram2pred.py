import nltk
from collections import Counter

corpus = []
one_gram = Counter()
bi_gram = Counter()
tri_gram = Counter()
data = open('data/final_corpus.txt').read().split('\n')
file = open('data/test_ngram.txt', 'w+')
for sent in data:
    sent = sent.split()
    one_gram.update(nltk.ngrams(sent, 1))
    bi_gram.update(nltk.ngrams(sent, 2))
    # tri_gram.update(nltk.ngrams(sent, 3))

# print(one_gram)

for word in one_gram:
    file.write(word[0] + '\n')

for word in bi_gram:
    file.write(" ".join(word) + '\n')
    # file.write(" ".join(l) + '\n')
