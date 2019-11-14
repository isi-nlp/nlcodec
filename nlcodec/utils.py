#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-11-12
from typing import List, Any, Iterable
import collections as coll
from nlcodec import log
from tqdm import tqdm

def make_n_grams(sent: List[Any], n):
    assert n > 0
    return [tuple(sent[i: i + n]) for i in range(len(sent) - n + 1)]


def make_n_grams_all(sents: Iterable[List[Any]], n):
    grams = coll.Counter()
    n_sent = 0
    for sent in tqdm(sents, mininterval=1, dynamic_ncols=True):
        grams.update(make_n_grams(sent, n))
        n_sent += 1
    log.info(f"Made {n}-grams: types={len(grams)}; tokens={sum(grams.values())}")
    return grams
