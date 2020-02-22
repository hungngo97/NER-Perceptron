from __future__ import division
from numpy.random import random
def dict_argmax(dct):
    """ Find argmax of a dict """
    return max(dct.keys(), key=lambda k: dct[k])


def viterbi(tag_transition_features, feature_values, output_vocab):
    N = len(feature_values)  
    
    # backpointer and value table represents all possible tags for each word position
    # Representation as a table of shape [Labels, Sentence_length] with number 
    # of rows is total possible labels and columns are each position in the sentence
    V = [{tag:None for tag in output_vocab} for t in range(N)]
    back = [{tag:None for tag in output_vocab} for t in range(N)]

    # Base case when it is at the first word, just weight * features
    V[0] = {k: feature_values[0][k] for k, v in V[0].items()}

    for i in range(1, len(V)):
        for k, v in V[i].items():
            # Try all possible paths
            probs = {key: feature_values[i][k] + V[i - 1][key] for key in V[i - 1].keys()}
            # probs = {key: tag_transition_features[(key, k)] + feature_values[i][k] + V[i - 1][key] for key in V[i - 1].keys()}
            V[i][k] = max(probs.values())
            back[i][k] = dict_argmax(probs)

    # Backtrace
    trace = [dict_argmax(V[len(V) - 1])]
    for i in range(1, len(back)):
        trace = [back[len(back) - i][trace[0]]] + trace
    return trace