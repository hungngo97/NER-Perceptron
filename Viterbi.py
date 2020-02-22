from __future__ import division
from numpy.random import random
from utils.math_utils import * 

def viterbi(A_factor, B_factors, output_vocab):

    N = len(B_factors)   # length of input sentence
    from utils.math_utils import dict_argmax


    # viterbi log-prob tables
    V = [{tag:None for tag in output_vocab} for t in range(N)]
    # backpointer tables
    # back[0] could be left empty. it will never be used.
    back = [{tag:None for tag in output_vocab} for t in range(N)]

    V[0] = {k: B_factors[0][k] for k, v in V[0].items()}

    for i in range(1, len(V)):
        for k, v in V[i].items():
            probs = {key: B_factors[i][k] + V[i - 1][key] for key in V[i - 1].keys()}
            
            # probs = {key: A_factor[(key, k)] + B_factors[i][k] + V[i - 1][key] for key in V[i - 1].keys()}
            V[i][k] = max(probs.values())
            back[i][k] = dict_argmax(probs)

    # Backtrace
    trace = [dict_argmax(V[len(V) - 1])]
    for i in range(1, len(back)):
        trace = [back[len(back) - i][trace[0]]] + trace

    return trace