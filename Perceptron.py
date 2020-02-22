from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint
import pickle
import re
from models.FeatureGenerator import ( FeatureGenerator )

##########################
# Globals
featureGenerator = FeatureGenerator()

import Viterbi
##########################
# Utilities

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.keys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.keys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def train(labels, examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    Student Implemented

    Train a perceptron. This is similar to the classifier perceptron training code
    but for the structured perceptron. Examples are now pairs of token and label
    sequences. The rest of the function arguments are the same as the arguments to
    the training algorithm for classifier perceptron.
    """
    OUTPUT_VOCAB = labels
    weights = defaultdict(float)

    S = defaultdict(float)
    t = 0

    def get_averaged_weights():
        """
        Student Implemented
        """
        return defaultdict(float, {feature: weights[feature] - ((1 / t) * val) for feature, val in S.items()})

    for pass_iteration in range(numpasses):
        print ("Training iteration %d" % pass_iteration)
        # Student Implemented

        for tokens,goldlabels in examples:
            g = dict_subtract(features_for_seq(tokens, goldlabels), features_for_seq(tokens, predict_seq(tokens, weights, OUTPUT_VOCAB)))
            for feature, gval in g.items():
                weights[feature] = weights[feature] + (stepsize * gval)
                S[feature] = S[feature] + ((t - 1) * stepsize * gval)
            t += 1

    return weights if not do_averaging else get_averaged_weights()

def predict_seq(tokens, weights, OUTPUT_VOCAB):
    """
    Student Implemented

    takes tokens and weights, calls viterbi and returns the most likely
    sequence of tags
    """
    # once you have Ascores and Bscores, could decode with
    # predlabels = greedy_decode(Ascores, Bscores, OUTPUT_VOCAB)
    if len(tokens) == 0:
        return []
    Ascores, Bscores = calc_factor_scores(tokens, weights, OUTPUT_VOCAB)
    return vit.viterbi(Ascores, Bscores, OUTPUT_VOCAB)

def greedy_decode(Ascores, Bscores, OUTPUT_VOCAB):
    """Left-to-right greedy decoding.  Uses transition feature for prevtag to curtag."""
    N=len(Bscores)
    if N==0: return []
    out = [None]*N
    out[0] = dict_argmax(Bscores[0])
    for t in range(1,N):
        tagscores = {tag: Bscores[t][tag] + Ascores[out[t-1], tag] for tag in OUTPUT_VOCAB}
        besttag = dict_argmax(tagscores)
        out[t] = besttag
    return out

def local_emission_features(t, tag, tokens):
    """
    Feature vector for the B_t(y) function
    t: an integer, index for a particular position
    tag: a hypothesized tag to go at this position
    tokens: the list of strings of all the word tokens in the sentence.
    Retruns a set of features.
    """
    features = {}
    features = featureGenerator.generate_local_feature(tokens, t, tag)
    return features

def features_for_seq(tokens, labelseq):
    """
    Student Implemented

    tokens: a list of tokens
    labelseq: a list of output labels
    The full f(x,y) function. Returns one big feature vector. This is similar
    to features_for_label in the classifier peceptron except here we aren't
    dealing with classification; instead, we are dealing with an entire
    sequence of output tags.

    This returns a feature vector represented as a dictionary.
    """
    total = defaultdict(float)
    for i in range(0, len(tokens)):
        for feat_name, feat_value in local_emission_features(i, labelseq[i], tokens).items():
            total[feat_name] += feat_value
        """
        if(i > 0):
            total["trans_%s_%s"% (labelseq[i - 1], labelseq[i])] += 1
        """
    return total

def calc_factor_scores(tokens, weights, OUTPUT_VOCAB):
    N = len(tokens)
    
    Ascores = {} 
    # Ascores = { (tag1,tag2): weights["trans_%s_%s"% (tag1, tag2)] for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Bscores = []
    for t in range(N):
        Bscores += [defaultdict(float)]
        for tag in OUTPUT_VOCAB:
            Bscores[t][tag] += dict_dotprod(weights, local_emission_features(t, tag, tokens))
    assert len(Bscores) == N
    return Ascores, Bscores
