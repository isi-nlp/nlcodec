#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-11-12

import argparse
import sys
import logging as log
from pathlib import Path
from nlcodec import load_scheme, EncoderScheme
from typing import Iterator
from nlcodec.utils import make_n_grams_all
import json
from tqdm import tqdm

log.basicConfig(level=log.INFO)
debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)


def encode_all(inp : Iterator[str], scheme: EncoderScheme, index=True):
    """
    :param inp: iterator sentences
    :param scheme:  encoder scheme
    :param index: True to get integer indexes, False to get string pieces
    :return: Iterator of Seqs, where seq is encoder per scheme as index
    """
    for line in tqdm(inp, mininterval=1, dynamic_ncols=True):
        line = line.strip()
        if index:
            yield scheme.encode(line)
        else:
            yield scheme.encode_str(line)


def main(inp, out, n, model_path):
    scheme = load_scheme(model_path)
    seqs = scheme.encode_parallel(inp)
    freqs = make_n_grams_all(seqs, n)
    freqs = sorted(freqs.items(), key=lambda x:x[1], reverse=True)
    for gram, freq in freqs:
        gram = list(gram)
        names = [scheme.table[g].name for g in gram]
        line = json.dumps([gram, freq, names], ensure_ascii=False)
        out.write(line + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Compute term frequencies for ")
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    p.add_argument('-m', '--model', dest='model_path', type=Path, required=True,
                   help="model aka vocabulary file")
    p.add_argument('-n', '--n', type=int, default=1, help='maximum n as in n-gram.')
    args = vars(p.parse_args())
    main(**args)
