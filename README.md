# Convert Neural LM to backing off n-gram

## How to Run

To train RNNLM, add the corpus to data folder

1. Train an n-gram LM using SRILM tool to generate sentences from corpus

   ```   
   ngram-count -text corpus.txt -order 6 -lm new_lm.arpa -vocab new_vocab 

   ngram -lm new_lm.arpa -gen [no_of_sentences] > gen.txt
   ```

2. To train RNNLM using above generated corpus as training data, run

    ```
    python train.py
    ```

    Evaluation result - Test perpexity value is ~52

## References

[1] Singh, Mittul & Oualil, Youssef & Klakow, Dietrich. (2017). \
Approximated and Domain-Adapted LSTM Language Models for First-Pass Decoding in Speech Recognition. \
2720-2724. 10.21437/Interspeech.2017-147. \

[2]  H. Adel, K. Kirchhoff, N. T. Vu, D. Telaar, and T. Schultz, \
“Comparing approaches to convert recurrent neural networks into backoff language models for efficient decoding,” \
in INTER-SPEECH 2014, 15th Annual Conference of the International Speech Communication Association, Singapore, September 14-18, 2014, 2014, pp.  51–655.
