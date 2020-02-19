#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-11-12
from typing import List, Any, Iterable, Dict, Tuple
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


def filter_types_coverage(types: Dict[str, int], coverage=1.0) -> Tuple[Dict[str, int], int]:
    assert  0 < coverage <= 1
    tot = sum(types.values())
    includes = {}
    cum = 0
    types  = sorted(types.items(), key=lambda x: x[1], reverse=True)
    for t, f in types:
        cum += f / tot
        includes[t] = f
        if cum >= coverage:
            break
    log.info(f'Coverage={cum:g}; requested={coverage:g}')
    excludes = {ch: ct for ch, ct in types if ch not in includes}
    unk_count = sum(excludes.values())
    log.warning(f'UNKed total toks:{unk_count} types={len(excludes)} from types:{excludes}')
    return includes, unk_count