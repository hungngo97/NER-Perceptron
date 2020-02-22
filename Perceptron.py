from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint
import pickle
import re
from models.FeatureGenerator import ( FeatureGenerator )
from utils.math_utils import *
import Viterbi

featureGenerator = FeatureGenerator()
def train(labels, examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    OUTPUT_VOCAB = labels
    weights = defaultdict(float)

    S = defaultdict(float)
    t = 0

    def get_averaged_weights():
        return defaultdict(float, {feature: weights[feature] - ((1 / t) * val) for feature, val in S.items()})

    for pass_iteration in range(numpasses):
        print ("Training iteration %d" % pass_iteration)

        for tokens,goldlabels in examples:
            g = dict_subtract(features_for_seq(tokens, goldlabels), features_for_seq(tokens, predict_seq(tokens, weights, OUTPUT_VOCAB)))
            for feature, gval in g.items():
                weights[feature] = weights[feature] + (stepsize * gval)
                S[feature] = S[feature] + ((t - 1) * stepsize * gval)
            t += 1

    return weights if not do_averaging else get_averaged_weights()

def predict_seq(tokens, weights, OUTPUT_VOCAB):
    if len(tokens) == 0:
        return []
    Ascores, Bscores = calc_factor_scores(tokens, weights, OUTPUT_VOCAB)
    return Viterbi.viterbi(Ascores, Bscores, OUTPUT_VOCAB)

def greedy_decode(Ascores, Bscores, OUTPUT_VOCAB):
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
    features = {}
    features = featureGenerator.generate_local_feature(tokens, t, tag)
    return features

def features_for_seq(tokens, labelseq):
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
