#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25

import collections as coll
import copy
import time
from typing import List, Dict, Tuple, Union, Iterator, Set

from nlcodec import log, DEF_MIN_CO_EV
from nlcodec.codec import Type, Level, Reseved
from nlcodec.dstruct import LnNode, MaxHeap
from nlcodec.utils import max_RSS
from tqdm import tqdm

Codes = Dict[int, Tuple[int, ...]]
Seq = List[int]
Bigram = Tuple[int, int]


class BPELearn:
    """
    The core BPE learning algorithm
    fast implementation using linked lists
    Note: this implementation takes relatively more RAM; and that is okay for my usecase
    # TODO: write this in c++ or rust and bind it here
    """
    space_tok = Reseved.SPACE_TOK[0]
    unk_tok = Reseved.UNK_TOK[0]

    def __init__(self, seqs: Iterator[Union[Seq, Tuple[Seq, int]]], vocab: List[Type]):
        # Check one to one map: type.name <-> idx
        assert len(set(v.idx for v in vocab)) == len(set(v.name for v in vocab))
        for i, v in enumerate(vocab):
            assert i == v.idx
        self.vocab = vocab
        self.uni: Dict[int, int] = coll.defaultdict(int)  # term freq ; unigrams
        self.bi: Dict[Bigram, int] = coll.defaultdict(int)  # bigram frequencies

        # Bigram to sequence references
        self.bi_ixs: Dict[Bigram, Set[LnNode]] = coll.defaultdict(set)

        self.create_index(seqs)
        self.validate_index()

    def create_index(self, seqs):
        log.info("Going to build corpus stats index; This might take lot of time and memory")
        n_seqs, n_ignored, n_replaced, bar_msg = 0, 0, 0, ''
        with tqdm(enumerate(seqs), unit='seqs', dynamic_ncols=True, mininterval=1) as data_bar:
            for idx, seq in data_bar:
                freq = 1  # default = 1 freq
                if isinstance(seq, tuple):  # if freq is available
                    seq, freq = seq

                n_seqs += 1
                if idx == 0:  # basic validation
                    assert isinstance(seq, list)  # first sequence, tokenized
                    assert isinstance(seq[0], int)  # sequence's item, should be an int or codepoint
                if not seq:
                    log.warning(f"Skipping empty sequence at idx {idx + 1}")
                    continue

                nodes = LnNode.from_seq(seq, freq=freq)
                assert len(seq) == len(nodes)

                for i in range(len(seq) - 1):  # the last position left out
                    bigm = (seq[i], seq[i + 1])
                    self.bi[bigm] += freq
                    assert nodes[i] not in self.bi_ixs[bigm]
                    self.bi_ixs[bigm].add(nodes[i])  # bigm found at node i
                    self.uni[seq[i]] += freq
                self.uni[seq[-1]] += freq  # the last unigram count; not covered in the above loop
                bar_msg = f'MaxRSS={max_RSS()[1]}'
                data_bar.set_postfix_str(bar_msg, refresh=False)
        log.info(f"Created index; {bar_msg}")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def validate_index(self):
        """
        Call this any time to check if the index of uni bi bi_ixs are valid.
        Raises exception on invalid index
        :return:
        """
        # This is code doesnt work with fast but new dirty heap updates
        max_code = max(self.uni)
        max_idx = max(t.idx for t in self.vocab)
        if not (max_code < self.vocab_size and max_code <= max_idx):
            raise ValueError(
                f'Vocab size is {self.vocab_size}, but max_code is {max_code}; max_idx={max_idx}')
        if not len(self.bi) == len(self.bi_ixs):
            raise ValueError(f"|bi|={len(self.bi)} and |bi_idxs|={len(self.bi_ixs)} are not same")
        for bigm, freq in self.bi.items():
            if not freq >= 0:
                raise ValueError(f"{bigm} has freq {freq}; expected positive val")
            if not bigm in self.bi_ixs:
                raise ValueError(f"{bigm} exists in bi but not in bi_ixs")
            idx_freq = sum(n.freq for n in self.bi_ixs[bigm])
            if not freq == idx_freq:
                raise ValueError(
                    f"{bigm} has freq={freq} bi but has {idx_freq} bi_ixs refs")
            # less than unigram freqs
            if not freq <= self.uni[bigm[0]]:
                raise ValueError(f"{bigm} has freq={freq} bi but {bigm[0]} has {self.uni[bigm[0]]}")
            if not freq <= self.uni[bigm[1]]:
                raise ValueError(f"{bigm} has freq={freq} bi but {bigm[1]} has {self.uni[bigm[1]]}")
        for uni, freq in self.uni.items():
            if not freq >= 0:
                raise ValueError(f"{uni} has freq={freq}; expected positive value")
        log.info(f"Index is valid")

    def learn_codes(self, n_merges: int, min_co_evidence, code_level: int,
                    log_every=2) -> List[Type]:
        """
        :param n_merges: how many more merges
        :param min_co_evidence: min evidence (co-occurrence frequency);
         causes early stop upon failure
        :param code_level: what level to use for new code types created during merge
            for instance level=1 for word bpe; level=2 for seq bpe
        :param log_every: delay, in seconds between logs
        :return:
        """
        uni, bi_ixs = self.uni, self.bi_ixs
        heap = MaxHeap(self.bi)
        heap_dirty = coll.defaultdict(int)  # subtractions aren't updated in max-heap, they are here
        vocab = self.vocab
        last_log_t = time.time()
        log.info(f"logs every {log_every} seconds")
        for i in range(n_merges):
            # Using MaxHeap for faster lookup of max. But heap gets a bit dirty, so a bit of cleanup
            max_pair, pair_freq = heap.pop()
            while max_pair in heap_dirty:  # clean all max [airs until a clean value
                freq_update = heap_dirty.pop(max_pair)
                assert freq_update < 0  # only decrements are valid. increments make this wrong
                corr_freq = pair_freq + freq_update  # correct value
                assert corr_freq >= 0, f'{max_pair}:{pair_freq}, Δ={freq_update} = {corr_freq}'
                if corr_freq > 0:  # exclude zero count
                    heap.push(max_pair, corr_freq)
                max_pair, pair_freq = heap.pop()

            # here the  actual loop begins
            if pair_freq < min_co_evidence:
                log.warning(f"Early stop; max evidence found is {pair_freq} "
                            f"but min required is {min_co_evidence}")
                break

            new_type_idx = len(vocab)
            a, b = max_pair
            if time.time() - last_log_t >= log_every:
                log.info(f"{(100 * i / n_merges):.2f}% :: {new_type_idx} || {a:4}:{uni[a]:5}"
                         f" || {b:4}:{uni[b]:5} || {pair_freq:,} || {vocab[a].name} {vocab[b].name}")
                last_log_t = time.time()

            # code -> bigram   (flatten out bigram;  resolve interim codes
            new_type = Type(vocab[a].name + vocab[b].name, idx=new_type_idx, freq=pair_freq,
                            level=code_level, kids=(vocab[a], vocab[b]))
            vocab.append(new_type)

            # updates: update bigram and unigram counts
            uni[new_type_idx] = pair_freq  # this bigram is now a new unigram
            # unigram counts drop ; since some of their bigrams are removed
            uni[a] -= pair_freq
            uni[b] -= pair_freq
            heap_deltas = coll.defaultdict(int)
            update_nodes = bi_ixs.pop(max_pair)  # also removed from bi_ixs
            for node in update_nodes:
                # -- x a b y --
                x_node, b_node = node.left, node.right
                if node.is_unlinked or (a == b and new_type.idx in (node.val, b_node.val)):
                    # this happens in the cases like "x a a a a y"
                    uni[a] += node.freq
                    uni[b] += node.freq
                    uni[new_type.idx] -= node.freq
                    continue

                y_node = b_node.right
                dirty = node.val != a or b_node.val != b  # check that the linked list is proper
                if dirty:
                    log.warning(f'Expected {a, b} but found {node.val, b_node.val}'
                                f'\n {node, b_node}'
                                f'\n--{vocab[a].signature()} =='
                                f' {vocab[node.val].signature() if node.val != a else "OK"}'
                                f'\n--{vocab[b].signature()} =='
                                f' {vocab[b_node.val].signature() if b_node.val != b else "OK"}')
                    log.warning(f"a={a}, b={b} || a_node={node}, b_node={b_node}")
                assert not dirty
                assert node.freq == b_node.freq

                # update : x a b y => x R y
                b_node.delete(unlink=True)  # delete() takes care of linking a → y and a ← y
                new_node = node  # reuse a node as new_node/R
                new_node.val = new_type_idx  # reuse a as new_node/R
                # Note: the above edits to a and b nodes do-not/should-not change __hash__

                if x_node:
                    # remove (x_node_val, a) from bi and bi_ixs
                    heap_deltas[(x_node.val, a)] -= x_node.freq
                    if bi_ixs.get((x_node.val, a)):
                        # not sure why 'if' needed here;
                        bi_ixs[(x_node.val, a)].remove(x_node)

                    # add (x_node_val, R) to bi and bi_ixs
                    heap_deltas[(x_node.val, new_type_idx)] += x_node.freq
                    bi_ixs[(x_node.val, new_type_idx)].add(x_node)
                if y_node:
                    # remove (b, y_node.val) from bi and bi_ixs
                    heap_deltas[(b, y_node.val)] -= b_node.freq
                    if bi_ixs.get((b, y_node.val)):
                        # not sure why 'if' needed here;
                        bi_ixs[(b, y_node.val)].remove(b_node)

                    # add (R, y_node.val) to bi and bi_ixs
                    heap_deltas[(new_type_idx, y_node.val)] += b_node.freq
                    bi_ixs[(new_type_idx, y_node.val)].add(new_node)

            # however; the counts shouldn't go negative
            assert uni[a] >= 0
            assert uni[b] >= 0

            for pair, delta in heap_deltas.items():
                if delta > 0:  # these are new insertions, and they can go directly to heap
                    assert new_type_idx in pair
                    heap.push(pair, delta)
                elif delta < 0:  # one of those subtractions, which cant be directly updated
                    assert new_type_idx not in pair
                    heap_dirty[pair] += delta

        return vocab

    @classmethod
    def prepare_word(cls, word):
        # mark ending of sequences
        # # TODO: check:  looks like sentence piece adds at the beginning
        # subword-nmt (senrich et al 2016) did </w> at the end;
        # 0.2 of subword-nmt puts last char and </w> together
        return word + cls.space_tok

    @classmethod
    def _make_idxs(cls, voc_idx: Dict[str, int], term_freqs: Dict[str, int]) \
            -> Iterator[Tuple[Seq, int]]:
        """Convert character sequences to char indexed seqs"""
        unk_idx = voc_idx[cls.unk_tok]
        for word, freq in term_freqs.items():
            if word in voc_idx:
                res = [voc_idx[word]]
            else:
                res = [voc_idx.get(ch, unk_idx) for ch in word]
            yield res, freq

    @classmethod
    def _learn_codes(cls, term_freqs: Dict[str, int], vocab: List[Type], vocab_size: int,
                     init_list: List[str] = None,
                     min_co_evidence: int = DEF_MIN_CO_EV) -> List[Type]:
        """
        :param term_freqs: words types and frequencies
        :param vocab: initial vocab; usually reserved and alphabet
        :param vocab_size: desired vocabulary size
        :param init_list: any reserved words
        :return: List[str] word pieces
        """

        vocab = copy.copy(vocab)
        log.info(f"Found {len(term_freqs)} types before splitting; initial vocab {len(vocab)}")
        if init_list:
            log.info(f'Adding {len(init_list)} types to the initial vocab')
            assert not any(' ' in w for w in init_list), 'spaces not allowed in init_list words'
            vocab += [Type(cls.prepare_word(w), level=Level.user, idx=idx, freq=0)
                      for idx, w in enumerate(init_list, start=len(vocab))]

        rev_idx: Dict[str, int] = {word.name: word.idx for word in vocab}
        assert len(rev_idx) == len(vocab)  # one to one map
        assert vocab_size > len(vocab), f'vocab_size={vocab_size} is too small;' \
                                        f' found {len(vocab)} in the init vocab! Set a value larger than {len(vocab)}'

        seqs_freqs = cls._make_idxs(rev_idx, term_freqs)
        learner = BPELearn(seqs_freqs, vocab=vocab)
        final_vocab = learner.learn_codes(n_merges=vocab_size - len(vocab),
                                          min_co_evidence=min_co_evidence,
                                          code_level=Level.subword)
        return final_vocab

    @classmethod
    def learn_subwords(cls, term_freqs: Dict[str, int], vocab_size: int,
                       min_co_evidence: int = DEF_MIN_CO_EV,
                       init_vocab_factory=None) -> List[Type]:
        """
        :param term_freqs:
        :param vocab_size: final vocab size: reserved + chars + user_specified  + merges;
          special case, when `vocab_size=-1` the returned vocab will have just reserved + chars
        :param min_co_evidence: min co evidence for pair merges
        :param char_coverage: percentage of characters to be covered by inital char_freqs
        :param word_min_freq: words below this frequency will be excluded for learning BPE
        :return: List of Type
        """

        log.info(f"Total types: {len(term_freqs)}")
        term_freqs = {cls.prepare_word(word): freq for word, freq in term_freqs.items()}

        char_freqs = coll.defaultdict(int)
        for term, freq in term_freqs.items():
            for ch in term:
                char_freqs[ch] += freq
            """TODO: test this behavior; similar to subword-nmt v0.2
            for ch in term[:-2]:  # skip the last two: ending and the whitespace marker
                char_freqs[ch] += freq
            char_freqs[term[-2:]] += freq  # ending + whitespace marker go together as a single byte
            """
        init_vocab = init_vocab_factory(char_freqs)  # create an initial vocabulary of chars
        if vocab_size == -1:
            log.warning(f'Since vocab_size={vocab_size}; not going to do any L1 merges')
            log.info(f'Found initial vocab size of {len(init_vocab)}')
            return init_vocab

        return cls._learn_codes(term_freqs, init_vocab, min_co_evidence=min_co_evidence,
                                vocab_size=vocab_size)

    @classmethod
    def learn_subwords_from_corpus(cls, corpus: Iterator[str], **kwargs) -> List[Type]:
        """
        :param corpus: line iterator
        :param **kwargs : Refer learn_subwords() args
        """
        term_freq = coll.Counter(word for seq in corpus for word in seq.strip().split())
        return cls.learn_subwords(term_freq, **kwargs)
