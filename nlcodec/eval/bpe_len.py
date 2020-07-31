#!/usr/bin/env python
#
# tool to estimate how the mean sequence length shrink with bpe merges
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-11-28
import argparse
import collections as coll
import sys
from pathlib import Path
from typing import Iterator, Dict, Set, Optional, Union

import numpy as np
from tqdm import tqdm

from nlcodec import log, BPEScheme, Type, Level
from nlcodec.bpe import Bigram, LnNode
from nlcodec.utils import max_RSS

NDArray = np.ndarray


class BPECodecXt(BPEScheme):
    """
    Use for merging one by one for the sake of tracking
    """

    def __init__(self, path: Union[str,Path]):
        types, meta = Type.read_vocab(path)
        assert meta
        assert meta['max_level'] == Level.subword
        init_table = [t for t in types if t.level < Level.subword]
        self.merge_buffer = [t for t in types if t.level >= Level.subword]
        for m in self.merge_buffer:
            assert len(m.kids) == 2  # exactly two pieces

        super().__init__(table=init_table) # start with character
        self.root = self.make_vocab_prefix_trie(self.table)
        assert self.unk_idx
        log.info(f"initial table has {len(init_table)}; buffer has {len(self.merge_buffer)}")

    def merges_remaining(self):
        return len(self.merge_buffer)

    def __len__(self):
        return len(self.table)

    def total_types(self):
        return len(self) + self.merges_remaining()

    def peek_merge(self) -> Type:
        return self.merge_buffer[0]

    def merge(self) -> Type:
        assert self.merge_buffer
        merge = self.merge_buffer.pop(0)
        assert len(self.table) == len(self.idx_to_str) == merge.idx
        self.table.append(merge)
        self.idx_to_str.append(merge.name)
        self.str_to_idx[merge.name] = merge.idx
        node = self.root.get_node(idxs=merge.name, create_missing=True)
        node.name = merge.name
        node.data = merge
        return merge


class BpeTracker:
    """
    Tracker for tracking how lengths and imbalance change over BPE merges
    """
    def __init__(self, bpe: BPECodecXt, data: Iterator[str]):
        self.bpe = bpe
        self.uni: NDArray = np.zeros(bpe.total_types(), dtype=np.long) # term freq ; unigrams

        # Bigram to sequence references
        self.bi_ixs: Dict[Bigram, Set[LnNode]] = coll.defaultdict(set)
        self.seq_lens: Optional[NDArray] = None
        self.create_index(data)

    def create_index(self, data):
        """
        1. encoder all : List[str] -> List[List[Int]]
        2. Count unigram freqs => uni
              2a) measure imbalance
              2b) measure seq lens
        3. Index all bigrams:: bi_ix (a,b) -> Node 
           Oh, yeah, convert all List[Int] to Doubly Linked List
           Create seq_len: List[int] each node should have an integer index of sequence in corpus
           
        4. ask bpe for next merge. gives (a,b) -> t
            a) count how many bigrams in index. 
                insert uni[t] = count
                reduce  uni[a], uni[b] by that count
            b) loop through each indexed node 
               remove b. update a;  update links :: x a b y => x t y
               insert bi_ix[x, t] and bi_ix[t, y]
               using seq number stored on node, decrement length by 1. assert len >= 1
            c) measure imbalance, and avg length
        """
        log.info("Encoding and creating index")
        enc_data  = self.bpe.encode_parallel(data)
        uni: Dict[int, int] = coll.defaultdict(int)
        bi_ixs: Dict[Bigram, Set[LnNode]] = coll.defaultdict(set)
        seq_lens = []
        with tqdm(enumerate(enc_data)) as data_bar:
            for idx, seq in data_bar:
                seq = LnNode.from_seq(seq, data=idx)
                seq_lens.append(len(seq))
                for node in seq:
                    uni[node.val] += 1
                for node in seq[:-1]:
                    bi_ixs[(node.val, node.right.val)].add(node)
                bar_msg = f'MaxRSS={max_RSS()[1]}'
                data_bar.set_postfix_str(bar_msg, refresh=False)

        self.seq_lens = np.array(seq_lens)
        log.info(f"Found {len(self.seq_lens)} sentences")
        for idx, freq in uni.items():
            self.uni[idx] = freq
        self.bi_ixs = bi_ixs

    @classmethod
    def earth_mov_dist(cls, stats: NDArray):
        k = len(stats)
        total = np.sum(stats)
        probs = stats / total
        diffs = np.abs(probs -  (1/k))
        return  0.5 * np.sum(diffs)

    def mean_seq_len(self):
        return np.mean(self.seq_lens)

    def track_merges(self, do_log=False):
        uni = self.uni
        bi_ixs = self.bi_ixs
        seq_lens = self.seq_lens
        res = []
        while self.bpe.merges_remaining() > 0:
            """
            ask bpe for next merge. gives (a,b) -> t
            a) count how many bigrams in index. 
                insert uni[t] = count
                reduce  uni[a], uni[b] by that count
            b) loop through each indexed node 
               remove b. update a;  update links :: x a b y => x t y
               insert bi_ix[x, t] and bi_ix[t, y]
               using seq number stored on node, decrement length by 1. assert len >= 1
            c) measure imbalance, and avg length
            """
            k = len(self.bpe)
            emd = self.earth_mov_dist(self.uni[:k])
            mu_len = self.mean_seq_len()
            res.append((k, emd, mu_len))
            if do_log:
                log.info(f"{k}: imbalance={emd:g} mean_seq_len={mu_len:g}."
                         f"Next: {self.bpe.peek_merge().signature()}")
            new_type = self.bpe.merge()
            a_type, b_type = new_type.kids
            a, b = a_type.idx, b_type.idx
            update_nodes = bi_ixs.pop((a, b), None)
            if update_nodes:
                freq = sum(u.freq for u in update_nodes)
                uni[new_type.idx] = freq
                uni[a] -= freq
                uni[b] -= freq
                for u in update_nodes:
                    b_node = u.right
                    if u.is_unlinked or (a == b and new_type.idx in (u.val, b_node.val)):
                        # happens due to repeats like X A A A A Y
                        uni[new_type.idx] -= u.freq
                        uni[a] += u.freq
                        uni[b] -= u.freq
                        continue
                    assert u.val == a
                    assert b_node.val == b, f'expected {a, b} found {u.val, b_node.val} at {b_node}'
                    u.val = new_type.idx
                    b_node.delete(unlink=True)

                    seq_idx = u.data
                    seq_lens[seq_idx] -= 1
                    assert seq_lens[seq_idx] > 0

                    if u.left:
                        if  bi_ixs.get((u.left.val, a)):
                            bi_ixs[(u.left.val, a)].remove(u.left)
                        bi_ixs[(u.left.val, u.val)].add(u.left)
                    if u.right:
                        if bi_ixs.get((b, u.right.val)):
                            bi_ixs[(b, u.right.val)].remove(b_node)
                        bi_ixs[(u.val, u.right.val)].add(u)
        k = len(self.bpe)
        emd = self.earth_mov_dist(self.uni[:k])
        mu_len = self.mean_seq_len()
        res.append((k, emd, mu_len))
        log.info(f"{k}: imbalance={emd:g} mean_seq_len={mu_len:g}.")
        return res


def estimate(codec_path, data):
    bpe = BPECodecXt(codec_path)
    tracker = BpeTracker(bpe, data)
    tracker.track_merges(do_log=True)

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