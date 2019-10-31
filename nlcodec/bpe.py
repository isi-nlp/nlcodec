#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25

from typing import List, Dict, Tuple, Union, Iterator, Set
import collections as coll
import resource
import sys
import copy
from tqdm import tqdm

from nlcodec import log, LnNode, Type, Level, Reseved

Codes = Dict[int, Tuple[int, ...]]
Seq = List[int]
Bigram = Tuple[int, int]

DEF_MIN_CO_EV = 5
DEF_CHAR_MIN_FREQ = 20  # minimum times a char should be seen to be included in l1 init vocab
DEF_WORD_MIN_FREQ = 1   # minimum times a word should exist to be used for l1 vocab


def max_RSS(who=resource.RUSAGE_SELF) -> Tuple[int, str]:
    """Gets memory usage of current process, maximum so far.
    Maximum so far, since the system call API doesnt provide "current"
    :returns (int, str)
       int is a value from getrusage().ru_maxrss
       str is human friendly value (best attempt to add right units)
    """
    mem = resource.getrusage(who).ru_maxrss
    h_mem = mem
    if 'darwin' in sys.platform:  # "man getrusage 2" says we get bytes
        h_mem /= 10 ** 3  # bytes to kilo
    unit = 'KB'
    if h_mem >= 10 ** 3:
        h_mem /= 10 ** 3  # kilo to mega
        unit = 'MB'
    return mem, f'{int(h_mem)}{unit}'


class BPELearn:
    """
    The core BPE learning algorithm
    fast implementation using linked lists
    Note: this implementation takes relatively more RAM; and that is okay for my usecase
    # TODO: write this in c++ or rust and bind it here
    """
    space_tok = Reseved.SPACE_TOK[0]
    unk_tok = Reseved.UNK_TOK[0]

    def __init__(self, seqs: Iterator[Union[Seq, Tuple[Seq, int]]], vocab: List[Type],
                 troubles='replace'):
        assert troubles in ['ignore', 'replace']
        # Check one to one map: type.name <-> idx
        assert len(set(v.idx for v in vocab)) == len(set(v.name for v in vocab))
        for i, v in enumerate(vocab):
            assert i == v.idx
        self.vocab = vocab
        self.uni: Dict[int, int] = coll.defaultdict(int)  # term freq ; unigrams
        self.bi: Dict[Bigram, int] = coll.defaultdict(int)  # bigram frequencies

        # Bigram to sequence references
        self.bi_ixs: Dict[Bigram, Set[LnNode]] = coll.defaultdict(set)

        self.create_index(seqs, troubles)
        self.validate_index()

    def create_index(self, seqs, troubles):
        log.info("Going to build corpus stats index; This might take lot of time and memory")
        n_seqs, n_ignored, n_replaced, bar_msg = 0, 0, 0, ''
        with tqdm(enumerate(seqs), unit='seqs', dynamic_ncols=True) as data_bar:
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

                if self._is_troublesome_seq(seq):
                    if troubles == 'ignore':
                        n_ignored += 1
                        continue
                    elif troubles == 'replace':
                        seq = self._replace_trouble(seq)
                        n_replaced += 1
                    else:
                        raise Exception('This should not be happening!')

                nodes = LnNode.from_seq(seq, freq=freq)
                assert len(seq) == len(nodes)

                for i in range(len(seq) - 1):  # the last position left out
                    bigm = (seq[i], seq[i + 1])
                    self.bi[bigm] += freq
                    assert nodes[i] not in self.bi_ixs[bigm]
                    self.bi_ixs[bigm].add(nodes[i])  # bigm found at node i
                    self.uni[seq[i]] += freq
                self.uni[seq[-1]] += freq  # the last unigram count; not covered in the above loop
                err_msg = f'Replaced={n_replaced}' if troubles == 'replace' \
                    else f'Ignored={n_ignored}'
                bar_msg = f'Seqs: Total={n_seqs} {err_msg}; MaxRSS={max_RSS()[1]}'
                data_bar.set_postfix_str(bar_msg)
        log.info(f"Created index; {bar_msg}")

    def _is_troublesome_seq(self, seq) -> bool:
        for i in range(2, len(seq)):
            """
            repetitions are bad; for example see below
            case1: seq= x 1 1 y  ; uni={x:1, 1:2, y:1}; bi= {(x,1):1, (1,1)=1, (1, y)=1}
            case2:, seq= x 1 1 1 y; uni={x:1, 1:3, y:1}; bi= {(x,1):1, (1,1)=2, (1, y)=1}
            case3, seq= x 1 1 1 1 y; uni={x:1, 1:4, y:1}; bi= {(x,1):1, (1,1)=3, (1, y)=1}
            => case1 is okay;  uni[1] -= bi[(1,1)] two times as usual
            => case2 is bad;   uni[1] -= bi[(1,1)] two times as is a mess up
            => case3 or longer is really a mess up; total mess up of replacements
            """
            # three or more consecutive same code points --> trouble!
            if seq[i] == seq[i - 1] == seq[i - 2]:
                return True
        return False

    def _replace_trouble(self, seq: List) -> List:
        # edit sequences like a111b --> a11b
        # ie. replace sequence of three or more repeated bytes into at most 2
        buffer = [1] * len(seq)
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                buffer[i] = buffer[i - 1] + 1
        res = [x for x, flag in zip(seq, buffer) if flag <= 2]
        return res

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def validate_index(self):
        """
        Call this any time to check if the index of uni bi bi_ixs are valid.
        Raises exception on invalid index
        :return:
        """
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

    def learn_codes(self, n_merges: int, min_co_evidence, code_level: int) -> List[Type]:
        """
        :param n_merges: how many more merges
        :param min_co_evidence: min evidence (co-occurrence frequency);
         causes early stop upon failure
        :param code_level: what level to use for new code types created during merge
            for instance level=1 for word bpe; level=2 for seq bpe
        :return:
        """
        bi, uni, bi_ixs = self.bi, self.uni, self.bi_ixs
        vocab = self.vocab
        new_code = self.vocab_size - 1  # -1 because the loop starts with an increment
        for i in range(n_merges):
            new_code += 1
            a, b = max_pair = max(bi, key=bi.get)  # get the max freq bigram
            pair_freq = bi[max_pair]
            if pair_freq < min_co_evidence:
                log.warning(f"Early stop; max evidence found is {pair_freq} "
                            f"but min required is {min_co_evidence}")
                break
            log.info(f"{(100 * i / n_merges):.2f}% :: {new_code} || {a:4}:{uni[a]:5}"
                     f" || {b:4}:{uni[b]:5} || {pair_freq:,} || {vocab[a].name} {vocab[b].name}")

            # code -> bigram   (flatten out bigram;  resolve interim codes
            new_type = Type(vocab[a].name + vocab[b].name, idx=new_code, freq=pair_freq,
                            level=code_level, kids=(vocab[a], vocab[b]))
            assert len(vocab) == new_type.idx
            vocab.append(new_type)

            # updates: update bigram and unigram counts
            del bi[max_pair]  # remove it; it is no longer a bigram
            uni[new_code] = pair_freq  # this bigram is now a new unigram
            # unigram counts drop ; since some of their bigrams are removed
            uni[a] -= pair_freq
            uni[b] -= pair_freq
            # however; the counts shouldn't go negative
            assert uni[a] >= 0
            assert uni[b] >= 0
            update_nodes = bi_ixs.pop(max_pair)  # also removed from bi_ixs
            for node in update_nodes:
                a_node, b_node = node, node.right
                dirty = a_node.val != a or b_node.val != b  # check that the linked list is proper
                if dirty:
                    self.validate_index()
                    log.warning(f"a={a}, b={b} || a_node={a_node}, b_node={b_node}")
                assert not dirty
                assert a_node.freq == b_node.freq
                x_node, y_node = a_node.left, b_node.right
                # update : x a b y => x R y
                b_node.delete()  # delete() takes care of linking a → y and a ← y
                new_node = a_node  # reuse a node as new_node/R
                new_node.val = new_code  # reuse a as new_node/R
                # Note: the above edits to a and b nodes do-not/should-not change __hash__

                if x_node and bi[(x_node.val, a)] > 0:
                    # remove (x_node_val, a) from bi and bi_ixs
                    bi[(x_node.val, a)] -= x_node.freq
                    assert bi[(x_node.val, a)] >= 0
                    bi_ixs[(x_node.val, a)].remove(x_node)
                    # add (x_node_val, R) to bi and bi_ixs
                    bi[(x_node.val, new_code)] += x_node.freq
                    bi_ixs[(x_node.val, new_code)].add(x_node)
                if y_node and bi[(b, y_node.val)] > 0:
                    # remove (b, y_node.val) from bi and bi_ixs
                    bi[(b, y_node.val)] -= b_node.freq
                    assert bi[(b, y_node.val)] >= 0
                    bi_ixs[(b, y_node.val)].remove(b_node)
                    # add (R, y_node.val) to bi and bi_ixs
                    bi[(new_code, y_node.val)] += b_node.freq
                    bi_ixs[(new_code, y_node.val)].add(new_node)

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
            [cls.prepare_word(w) for w in init_list]
            vocab += [Type(cls.prepare_word(w), level=Level.user, idx=idx, freq=0)
                      for idx, w in enumerate(init_list, start=len(vocab))]

        rev_idx: Dict[str, int] = {word.name: word.idx for word in vocab}
        assert len(rev_idx) == len(vocab)  # one to one map
        assert vocab_size > len(vocab), f'vocab_size={vocab_size} is too small;' \
            f' found {len(vocab)} in the init vocab!'

        seqs_freqs = cls._make_idxs(rev_idx, term_freqs)
        learner = BPELearn(seqs_freqs, vocab=vocab)
        final_vocab = learner.learn_codes(n_merges=vocab_size - len(vocab),
                                          min_co_evidence=min_co_evidence,
                                          code_level=Level.subword)
        return final_vocab

    @classmethod
    def learn_subwords(cls, term_freqs: Dict[str, int], vocab_size: int,
                       min_co_evidence: int = DEF_MIN_CO_EV,
                       char_min_freq=DEF_CHAR_MIN_FREQ,
                       word_min_freq=DEF_WORD_MIN_FREQ) -> List[Type]:
        """
        :param term_freqs:
        :param vocab_size: final vocab size: reserved + chars + user_specified  + merges;
          special case, when `vocab_size=-1` the returned vocab will have just reserved + chars
        :param min_co_evidence: min co evidence for pair merges
        :param char_min_freq: characters below this frequency will be unk'ed
        :param word_min_freq: words below this frequency will be excluded for learning BPE
        :return: List of Type
        """
        assert char_min_freq >= 1
        assert word_min_freq >= 1

        log.info(f"Total types: {len(term_freqs)}")
        term_freqs = {cls.prepare_word(word): freq for word, freq in term_freqs.items()
                      if freq >= word_min_freq}
        if word_min_freq > 1:
            log.info(f"Total types after min_freq >= {word_min_freq}: {len(term_freqs)}")

        alphabet = coll.defaultdict(int)
        for term, freq in term_freqs.items():
            for ch in term:
                alphabet[ch] += freq
            """TODO: test this behavior; similar to subword-nmt v0.2
            for ch in term[:-2]:  # skip the last two: ending and the whitespace marker
                alphabet[ch] += freq
            alphabet[term[-2:]] += freq  # ending + whitespace marker go together as a single byte
            """
        if char_min_freq > 1:
            includes = {ch: freq for ch, freq in alphabet.items() if freq >= char_min_freq}
            excludes = {ch: ct for ch, ct in alphabet.items() if ch not in includes}
            log.info(f'unked chars count:{sum(excludes.values())} from types:{excludes}')
            alphabet = includes
        else:
            log.info("Character coverage: full")

        init_vocab = Reseved.with_reserved_types()  # initial vocab with reserved toks
        [alphabet.pop(v.name, None) for v in init_vocab if v.name in alphabet]  # remove reserved ch
        alphabet = sorted(alphabet.items(), key=lambda x: x[1], reverse=True)  # high freq on top

        init_vocab += [Type(name, level=Level.char, idx=idx, freq=freq)
                       for idx, (name, freq) in enumerate(alphabet, start=len(init_vocab))]

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
