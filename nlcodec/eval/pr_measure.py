#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 2019-11-11

import argparse
import sys
import logging as log
import io
import collections as coll
from nlcodec.utils import make_n_grams, make_n_grams_all
from nlcodec import load_scheme, EncoderScheme
from pathlib import Path
import json

log.basicConfig(level=log.INFO)
debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)


def count_gram_recall(cands, refs, n=1):
    gram_recalls = coll.defaultdict(list)
    gram_precisions = coll.defaultdict(list)
    ref_gram_freqs = coll.defaultdict(int)
    cand_gram_freqs = coll.defaultdict(int)

    for cand, ref in zip(cands, refs):
        cand_grams = coll.Counter(make_n_grams(cand, n))
        ref_grams = coll.Counter(make_n_grams(ref, n))
        for ref_gram, ref_freq in ref_grams.items():
            assert ref_freq > 0
            cand_freq = cand_grams.get(ref_gram, 0)
            gram_recalls[ref_gram].append(min(cand_freq, ref_freq) / ref_freq)
            ref_gram_freqs[ref_gram] += ref_freq

        for cand_gram, cand_freq in cand_grams.items():
            assert cand_freq > 0
            ref_freq = ref_grams.get(cand_gram, 0)
            gram_precisions[cand_gram].append(min(cand_freq, ref_freq) / cand_freq)
            cand_gram_freqs[cand_gram] += cand_freq

    # average at the end; TODO: moving average
    gram_recalls = {gram: sum(recalls) / len(recalls) for gram, recalls in gram_recalls.items()}
    gram_precisions = {gram: sum(precs) / len(precs) for gram, precs in gram_precisions.items()}
    return gram_recalls, ref_gram_freqs, gram_precisions, cand_gram_freqs

def f1_measure(precison, recall):
    assert 0 <= precison <= 1
    assert 0 <= recall <= 1
    denr = precison + recall
    if denr == 0:
        return 0  # numerator 2*p*r is also zero
    return 2 * precison * recall / denr

def main(model_path, cands, refs, n, out, freqs=None):
    codec = load_scheme(model_path)
    cands, refs = list(cands), list(refs)
    assert len(cands) == len(refs), f'cands: {len(cands)} but refs: {len(refs)} lines'

    cands = list(codec.encode_parallel(cands))
    refs = list(codec.encode_parallel(refs))
    gram_recalls, ref_gram_freqs, gram_precisions, cand_gram_freqs = count_gram_recall(cands, refs)
    if freqs:
        log.info(f"Loading precomputed gram freqs from {freqs}")
        freqs = [json.loads(l.strip()) for l in freqs]
        gram_freqs = {tuple(g): f for g, f, name in freqs}

        # subset of grams that are found in reference
        gram_freqs = {g: f for g, f in gram_freqs.items() if g in ref_gram_freqs or g in cand_gram_freqs}

        # these grams were not found in training, but in there in refs => OOVs => freq=-1
        oov_grams = {g: -1 for g in ref_gram_freqs if g not in gram_freqs}
        log.info(f"{len(oov_grams)} grams were oov wrt to freqs => assigned freq = -1 ")
        gram_freqs.update(oov_grams)
    else:
        gram_freqs = ref_gram_freqs
    #print(gram_freqs.keys())
    new_grams = {cand_gram: freq for cand_gram, freq in cand_gram_freqs.items()
                   if cand_gram not in gram_freqs}

    if new_grams:
        msg = f'Found {len(new_grams)} grams that are not found in refs or --freqs'
        log.warning(msg)
        if n == 1:
            for ng, f in new_grams.items():
                ng = ng[0]
                log.error(f'Not found:\t{ng}\t{codec.idx_to_str[ng]}\t{f}')
            #raise Exception(msg)
        else:
            log.warning("TG, Come back and handle bigrams and above :)")

    gram_freqs = sorted(gram_freqs.items(), key=lambda t: t[1], reverse=True)
    out.write(f'Rank\tGram\tName\tRankF\tRefF\tCandF\tRecall\tPrecision\tF1\n')
    for i, (gram, rank_freq) in enumerate(gram_freqs):
        name = ','.join(codec.idx_to_str[g] for g in gram)
        idxs = ','.join(str(g) for g in gram)
        gram_recall = gram_recalls.get(gram, 0)
        gram_precision = gram_precisions.get(gram, 1) # should it be zero or one?
        f1 = f1_measure(gram_precision, gram_recall)
        ref_freq = ref_gram_freqs.get(gram, -1)
        cand_freq = cand_gram_freqs.get(gram, -1)
        out.write(f'{i+1}\t{idxs}\t{name}\t{rank_freq}\t{ref_freq}\t{cand_freq}\t{gram_recall:g}'
                  f'\t{gram_precision:g}\t{f1:g}\n')


def parse_args():
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore', newline='\n')
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Tool to compute Recall vs Frequency correlation.")
    p.add_argument('-c', '--cands', type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                   default=stdin, help='Candidate (aka output from NLG system) file')
    p.add_argument('-r', '--refs', type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                   default=stdin,
                   help='Reference (aka human label) file')
    p.add_argument('-f', '--freqs', type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                   help='precomputed freqs of grams on some other data (such as training) '
                        ' which should be used for ranking.'
                        ' If not given, --refs is used. This can be obtained from `termfreqs.py`')

    p.add_argument('-n', '--n', type=int, default=1, help='maximum n as in n-gram.')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=stdout,
                   help='Output file path to store the result.')
    p.add_argument('-m', '--model', dest='model_path', type=Path, required=True,
                   help="model aka vocabulary file")
    args = vars(p.parse_args())
    assert not (args['cands'] == args['refs'] == stdin), \
        'Only one of --refs and --cands can be read from STDIN'
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**args)
