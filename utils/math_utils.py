# -*- coding: utf-8 -*-
import math
from collections import defaultdict

def dict_argmax(dct):
    return max(dct.keys(), key=lambda k: dct[k])
def dict_subtract(vec1, vec2):
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    return max(dct.keys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    smaller = d1 if len(d1)<len(d2) else d2
    total = 0
    for key in smaller.keys():
        total += d1.get(key,0) * d2.get(key,0)
    return total
