#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 2019-11-12

import argparse
import collections as coll
import logging as log
import sys
from pathlib import Path
from typing import Iterator, Set, Dict, Tuple

from tqdm import tqdm

from nlcodec import load_scheme, WordScheme

log.basicConfig(level=log.INFO)
TermFreqs = Dict[str, int]


def term_freqs(inp : Iterator[str]) -> TermFreqs:
    tfs = coll.Counter()

    for line in tqdm(inp, mininterval=1, dynamic_ncols=True):
        toks = line.strip().split()
        tfs.update(toks)
    return  tfs

def partition_vocab_toks(inp : Iterator[str], vocab: Set[str]) -> Tuple[TermFreqs, TermFreqs]:
    tfs = term_freqs(inp)
    iv_stats = {t:f for t, f in tfs.items() if t in vocab}
    oov_stats = {t: f for t, f in tfs.items() if t not in vocab}
    return  iv_stats, oov_stats

def main(inp, model_path):
    codec = load_scheme(model_path)
    assert isinstance(codec, WordScheme)
    vocab = set(codec.str_to_idx.keys())
    iv_toks, oov_toks = partition_vocab_toks(inp, vocab)
    n_iv_types = len(iv_toks)
    n_iv_toks = sum(iv_toks.values())
    n_oov_types = len(oov_toks)
    n_oov_toks = sum(oov_toks.values())
    total_types = n_iv_types + n_oov_types
    total_toks = n_iv_toks + n_oov_toks

    print("*\tInVocab\tOOV")
    print(f"Types\t{n_iv_types}\t{n_oov_types}")
    print(f"Token Count\t{n_iv_toks}\t{n_oov_toks}")
    print(f"Token %\t{100*n_iv_toks/total_toks:.2f}\t{100*n_oov_toks/total_toks:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Compute term frequencies for ")
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-m', '--model', dest='model_path', type=Path, required=True,
                   help="model aka vocabulary file")
    args = vars(p.parse_args())
    main(**args)
