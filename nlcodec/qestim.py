#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25


import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Any

import numpy as np
from tqdm import tqdm

from nlcodec import EncoderScheme, log, load_scheme

NDArray = np.ndarray


@dataclass()
class CodecQuality:
    n_types: int
    imbalance: Any
    n_tokens: int
    mean_seq_len: float
    unk_count: int = 0


class QualityEstimator:

    def __init__(self, codec: EncoderScheme):
        self.codec = codec

    def estimate(self, lines: Iterator[str]) -> CodecQuality:
        seqs = (self.codec.encode(line) for line in lines)
        return self._estimate(seqs)

    @staticmethod
    def validate_distribution(distr):
        assert distr.dtype == np.float
        assert abs(np.sum(distr) - 1.0) < 1e-4  # sum to 1
        assert np.sum(distr < 0) == 0  # no negatives

    @classmethod
    def earth_mov_dist(cls, before: NDArray, after: NDArray, moving_cost=0.5, validate=True):
        if validate:
            cls.validate_distribution(before)
            cls.validate_distribution(after)
        assert len(before) == len(after)
        assert moving_cost > 0
        return  moving_cost * np.sum(np.abs(after - before))

    @classmethod
    def EMD_simple(cls, stats: NDArray, moving_cost=1):
        distr = stats / np.sum(stats)
        k = len(stats)
        assert moving_cost > 0
        return 0.5 * moving_cost * np.sum(np.abs(distr - 1/k))

    @classmethod
    def kl_div(self, base, distr):
        """
        Dkl (P || Q) should be read such that P is base from which Q's  divergence is computed
        Dkl (P || Q) =  ∑ P(x) log(P(x)/Q(x)) = - ∑ P(x) log(Q(x) / P(x))
        if base p is 0, then log(p/q) is not meaningful.
        if q is 0, we are in trouble; either log(0) = ∞ or log(1 / 0) which is division-by-zero
        so make sure q is never zero. Use JS Div which solves this
        :param base: P in DKL(P || Q)
        :param distr: Q in DKL(P || Q)
        :return: DKL(P || Q)
        """
        total = 0.0
        for px, qx in zip(base, distr):
            if px > 0:
                assert qx > 0
                total += px * np.log(px / qx)
        return total

    @classmethod
    def js_div(cls, base: NDArray, distr: NDArray):
        cls.validate_distribution(base)
        cls.validate_distribution(distr)
        assert len(base) == len(distr)

        middle = 0.5 * (base + distr)
        return 0.5 * (cls.kl_div(base=base, distr=middle) + cls.kl_div(base=distr, distr=middle))

    def _estimate(self, seqs: Iterator[List[int]]) -> CodecQuality:
        stats = [0] * self.codec.vocab_size
        # lengths = coll.defaultdict(int)
        n_seqs = 0
        for seq in tqdm(seqs):
            n_seqs += 1
            for tok in seq:
                stats[tok] += 1

        n_toks = sum(stats)
        mean_len = n_toks / n_seqs
        n_types = self.codec.vocab_size

        unk_count = stats[self.codec.unk_idx]

        # exclude the reserved types which dont surface in prediction
        #excludes = [typ for typ in self.codec.table if typ.is_reserved and stats[typ.idx] == 0]
        #n_types -= len(excludes)  # exclude reserved
        #for typ in reversed(excludes):  # delete from right
        #    log.info(f'Remove type from balance check: {typ}')
        #    del stats[typ.idx]
        n_toks = sum(stats)
        imb_measure = self.EMD_simple(stats=stats)
        return CodecQuality(n_types=n_types, imbalance=imb_measure, n_tokens=n_toks,
                            mean_seq_len=mean_len, unk_count=unk_count)


def estimate(codec_path, data):
    codec = load_scheme(codec_path)
    estimator = QualityEstimator(codec)
    estimation = estimator.estimate(data)
    print(estimation)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-c', '--codec', dest='codec_path', type=Path,
                   required=True, help='Path to codec vocabulary')
    p.add_argument('-d', '--data', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input data file path')
    return vars(p.parse_args())


def main():
    args = parse_args()
    estimate(**args)


if __name__ == '__main__':
    main()
