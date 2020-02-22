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
def train(labels, data, iterations):
    OUTPUT_VOCAB = labels
    weights = defaultdict(float)
    cum_weight = defaultdict(float)

    S = defaultdict(float)
    t = 0

    def get_averaged_weights():
        # return defaultdict(float,{feature: val / t for feature, val in cum_weight.items()} )
        return defaultdict(float, {feature: weights[feature] - ((1 / t) * val) for feature, val in S.items()})

    for i in range(iterations):
        print ("Training iteration %d" % i)
        for texts,label in data:
            diff = dict_subtract(generate_global_feature(texts, label), generate_global_feature(texts, predict_seq(texts, weights, OUTPUT_VOCAB)))
            for feature, val in diff.items():
                weights[feature] = weights[feature] + (val)
                S[feature] = S[feature] + ((t - 1) * val)
                cum_weight[feature] += weights[feature]
            t += 1

    return get_averaged_weights()

def predict_seq(texts, weights, OUTPUT_VOCAB):
    if len(texts) == 0:
        return []
    Ascores, Bscores = calc_feature_weight_value(texts, weights, OUTPUT_VOCAB)
    return Viterbi.viterbi(Ascores, Bscores, OUTPUT_VOCAB)

def generate_local_feature(t, tag, texts):
    features = {}
    features = featureGenerator.generate_local_feature(texts, t, tag)
    return features

def generate_global_feature(texts, labels):
    total = defaultdict(float)
    for i in range(0, len(texts)):
        for feat_name, feat_value in generate_local_feature(i, labels[i], texts).items():
            total[feat_name] += feat_value
        if(i > 0):
            total["trans_%s_%s"% (labels[i - 1], labels[i])] += 1
    return total

def calc_feature_weight_value(texts, weights, OUTPUT_VOCAB):
    N = len(texts)
    
    Ascores = {} 
    Ascores = { (tag1,tag2): weights["trans_%s_%s"% (tag1, tag2)] for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Bscores = []
    for t in range(N):
        Bscores += [defaultdict(float)]
        for tag in OUTPUT_VOCAB:
            Bscores[t][tag] += dict_dotprod(weights, generate_local_feature(t, tag, texts))
    return Ascores, Bscores
