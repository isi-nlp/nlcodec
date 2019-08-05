#!/usr/bin/env python
#
__author__ = "Thamme Gowda [tg@isi.edu]"
# Created: July 2019
#
# This software is Copyright © 2019 The University of Southern California. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research
# and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the
# above copyright notice, this paragraph and the following three paragraphs appear in all copies.

# Refer to LICENSE.txt file for full terms and conditions
#
import argparse
import sys
import logging as log
from pathlib import Path
import collections as coll
from typing import List, TextIO, Dict, Tuple, Union, Iterator, Optional, Set, TypeVar, Generic
import tqdm
import resource
from dataclasses import dataclass, field
import copy
import json
from datetime import datetime

__version__ = 0.1
__description__ = """Byte Pair Encoding++ is a tool for hierarchical merging of character streams 
into words and phrases."""
__epilog__ = f"""URL : https://github.com/thammegowda/bpepp """

log.basicConfig(level=log.INFO)
Codes = Dict[int, Tuple[int, ...]]
Seq = List[int]
Bigram = Tuple[int, int]

PAD_TOK = '<pad>', 0
UNK_TOK = '<unk>', 1  # unk = '⁇'  # U+2047  to make up some OOV characters
BOS_TOK = '<s>', 2
EOS_TOK = '</s>', 3
CLS_TOK = '<cls>', 4
SPACE_TOK = '▁', 5  # U+2581 same as google/sentencepiece

PAD_TOK_IDX = PAD_TOK[1]
UNK_TOK_IDX = UNK_TOK[1]
BOS_TOK_IDX = BOS_TOK[1]
EOS_TOK_IDX = EOS_TOK[1]
CLS_TOK_IDX = CLS_TOK[1]

RESERVED_TOKS = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK, CLS_TOK, SPACE_TOK]
N_RESERVED_TOKS = len(RESERVED_TOKS)

DEF_CHAR_COVERAGE = 0.9995
DEF_MIN_CO_EV = 5

"""
Some terminology ; in case I forget my own thoughts 😝
Approach: this is like hierarchical bype pair encoding (BPE) with two level compressions

level0: characters of the dataset 
level1 : expand word tokens as character seqs, do BPE to compress it
level2 : take level1 sequence of subwords and do BPE to compress it

Just for filling:
level-1: reserved toks <s> </s> <unk> etc which goes along side these above types (no role in BPE)
level0: user defined tokens; treat them just like level0 chars (dont further split them)  

The original paper, subword-nmt (Senrich et al) did BPE at level1 started from level0.

Didnt find any others who go to second level BPE yet (TODO: lit review, check check check)

Initially, I was planning to write a wrapper for sentencepiece (Kudo et al),
However I dropped the idea, bcoz integration became a mess; we like pretty code :)  
Maybe I will do that later if this idea is useful. Sentence piece is a nice library ❤. 
(However, bit more over engineered)
  
wish I was smart enough to write C/C++ code like sentencepiece or fastBPE,
 for now, I can gain some speed in python using fancy datastrutures  ;)  

Specifically using these to speed up this algorithm
1. Using doubly linked list to learn BPE codes 
2. using Prefix Trie to greedily apply BPE segmentation 

These data structures use so much more memory for internal indexes but thats okay for now
TODO: learn Rust/ modern C++
"""


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
        h_mem /= 10 ** 3  # bytes to mega
    unit = 'KB'
    if h_mem >= 10 ** 3:
        h_mem /= 10 ** 3  # kilo to mega
        unit = 'MB'
    return mem, f'{int(h_mem)}{unit}'


##### Data structures ######
@dataclass(repr=False)
class LnNode:  # doubly linked list node data structure; used for learning BPE
    val: int
    left: Optional['LnNode'] = None
    right: Optional['LnNode'] = None
    freq: int = 1

    def __eq__(self, other):
        # caution: these calls are recursive on left and right; cycles would cause infinite loop
        return (other.val == self.val and other.freq == self.freq and
                other.left == self.left and other.right == self.right)

    def __hash__(self):
        return id(self)  # quick and dirty hash; not sure how this mess if we use multiprocessing

    def delete(self):
        """
        deletes this node from the list
        :return:
        """
        x, y = self.left, self.right
        if x:  # right links : x → self → y  => x → y
            x.right = y
        if y:  # left links  : x ← self ← y  => x ← y
            y.left = x

    @classmethod
    def from_seq(cls, string: Union[str, List[int]], freq=1) -> List['LnNode']:
        """
        makes a doubly linked list from string
        :param string: input string (of integers recommended)
        :param freq: frequency of string in corpus (for scaling
        :return: List of Nodes, doubly linked to lefts and rights
        """
        nodes = [cls(ch, freq=freq) for ch in string]
        for i, n in enumerate(nodes):
            if i > 0:
                n.left = nodes[i - 1]
            if i + 1 < len(nodes):
                n.right = nodes[i + 1]
        return nodes

    def __repr__(self):
        lefts, rights = [], []

        cur = self.left
        while cur:
            lefts.append(str(cur.val))
            cur = cur.left
        cur = self.right
        while cur:
            rights.append(str(cur.val))
            cur = cur.right
        return ' '.join(reversed(lefts)) + f' *{self.val}* ' + ' '.join(rights)


T = TypeVar('T')  # T for index; Dont use I as in pep008
D = TypeVar('D')  # D for data


@dataclass()
class TrNode(Generic[T, D]):  # Trie Node or Tree Node
    idx: T
    name: Optional[str] = None
    data: Optional[D] = None
    parent: Optional['TrNode'] = None
    kids: Dict[T, 'TrNode'] = field(default_factory=dict)

    def get_node(self, idxs, create_missing: bool = True) -> 'TrNode[T, D]':
        if not idxs:
            return self
        if create_missing and idxs[0] not in self.kids:  # make one
            self.kids[idxs[0]] = TrNode(idx=idxs[0], parent=self)
        return self.kids[idxs[0]].get_node(idxs=idxs[1:], create_missing=create_missing)

    @property
    def is_root(self):
        return self.parent is None

    @property
    def n_kids(self):
        """Number of immediate children"""
        return len(self.kids)

    @property
    def is_leaf(self):
        return self.n_kids == 0

    @property
    def has_data(self):
        return self.data is not None

    @property
    def size(self):
        """Number of nodes in the tree at this node. Counts self and all the kids"""
        return 1 + sum(k.size for k in self.kids.values())

    @property
    def data_node_count(self):
        return (1 if self.has_data else 0) + sum(k.data_node_count for k in self.kids.values())


class Level:
    reserved = -1
    char = 0
    user = 0  # as of now,  user and char are indistinguishable; 0 means dont look inside
    subword = 1
    phrase = 2


@dataclass(frozen=True)
class Type:  # Type as in word type vs token
    name: str  # name of the type
    level: int  # in [-1, 0, 1, 2]
    idx: int  # idx for modeling integer sequence
    freq: int  # when available; from corpus
    kids: Optional[Tuple['Type', ...]] = None  # in case it was composed by sub pieces

    def format(self, delim='\t') -> str:
        cols = [str(self.idx), self.name, str(self.level), str(self.freq),
                ' '.join(str(k.idx) for k in self.kids) if self.kids else '']
        return delim.join(cols)

    @classmethod
    def parse(cls, line: str, vocab: List['Type'], delim='\t'):
        cols = line.strip().split(delim)
        idx, name, level, freq = cols[:4]
        idx, level, freq = int(idx), int(level), int(freq)
        kids = None
        if len(cols) > 4 and cols[4]:  # the last column maybe stripped out due to empty string
            kids = list(map(int, cols[4].split()))
            # kids shouldn't be referenced forward in index
            assert all(k < idx and k < len(vocab) for k in kids)
            kids = [vocab[k] for k in kids]
        return Type(name, level=level, idx=idx, freq=freq, kids=kids)

    @classmethod
    def with_reserved_types(cls) -> List['Type']:
        return [Type(tok, level=Level.reserved, idx=idx, freq=-1) for tok, idx in RESERVED_TOKS]


#########
class BpeCodec:
    unk_tok, unk_idx = UNK_TOK
    space_tok, space_idx = SPACE_TOK

    def __init__(self, vocab: Union[str, Path, TextIO, List[Type]]):
        """
        :param vocab: can be path string, Path object, opened file or List[Type]
        """
        if isinstance(vocab, str):
            vocab = Path(vocab)
        if isinstance(vocab, Path) or isinstance(vocab, TextIO):
            vocab = self.read_vocab(vocab)
        assert vocab and isinstance(vocab, List) and isinstance(vocab[0], Type)
        self.vocab: List[Type] = vocab
        self.root = self.make_vocab_prefix_trie(self.vocab)
        log.info(f"Vocab size={len(self.vocab)}; trie root has nodes={self.root.size}"
                 f" but data_nodes={self.root.data_node_count}")

    @classmethod
    def make_vocab_prefix_trie(cl, vocab: List[Type]):
        root: TrNode = TrNode[str, Type](idx='')
        for typ in vocab:
            node = root.get_node(idxs=typ.name, create_missing=True)
            node.name = typ.name
            node.data = typ
        assert not root.has_data  # root node is not data node
        return root

    @classmethod
    def write_vocab(cls, vocab: List[Type], out: Union[Path, TextIO]):
        wrtr = out.open('w', encoding='utf8', errors='ignore') if isinstance(out, Path) else out
        levels = dict(coll.Counter(v.level for v in vocab))
        meta = dict(total=len(vocab), version=__version__, levels=levels,
                    created=str(datetime.now()))
        meta = json.dumps(meta)
        wrtr.write(f"#{meta}\n")
        for i, item in enumerate(vocab):
            assert i == item.idx
            wrtr.write(item.format() + '\n')
        if wrtr != out:
            wrtr.close()
        log.info(f"Wrote {len(vocab)} to {wrtr.name}")

    @classmethod
    def read_vocab(cls, inp: Union[Path, TextIO]) -> List[Type]:
        rdr = inp.open('r', encoding='utf8', errors='ignore') if isinstance(inp, Path) else inp
        lines = list(l.strip() for l in rdr)
        if lines[0].startswith("#"):
            meta = lines.pop(0)  # metadata such as version; not used as of now

        # noinspection PyTypeChecker
        vocab: List[Type] = [None] * len(lines)
        for i, line in enumerate(lines):
            v = Type.parse(line=line.rstrip('\n'), vocab=vocab)
            assert v.idx == i
            vocab[i] = v
        if rdr != inp:
            rdr.close()
        log.info(f"read {len(vocab)} types from {rdr.name}")
        return vocab

    def encode(self, seq: str, pieces=False) -> Union[List[int], List[str]]:

        res: List[int] = []
        """
          T  h  i  s  _  X  i  s  _ 
          T  h*  i  s  _* X i  s  _*     # * is data node
        ↑prev_node
        Say vocab has: Th, This_, is_, Xmas_, This_Xmas_ ;; 'X' is unk but Xmas_ is valid 
          T  h  i  s  _*  X  i  s  _*    # Greedy; This_* instead of Th*
          T  h  i  s  _*  ??  i  s  _*   # Try for This_Xmas_; not possible? Backup and Turn 'X' as unk  
          [This_, ??, is_]
        """
        # Note: could have been done recursively; but iterative is stack efficient
        # TODO: this can be improved; dont make too many sub string objs => iterate over char array
        data_node, data_idx = None, -1
        prev_node, idx = self.root, 0
        while seq and idx < len(seq) + 1:  # +1 extra loop to handle the buffer at the end
            if prev_node.has_data:
                data_node, data_idx = prev_node, idx

            if idx < len(seq) and seq[idx] in prev_node.kids:  # keep growing, i.e. advance
                prev_node = prev_node.kids[seq[idx]]
                idx += 1
            else:
                if data_node:  # the last seen data node exists
                    res.append(data_node.data.idx)
                    seq = seq[data_idx:]  # reset to the successors of last seen data node
                else:  # there was no data node seen; backup and make unk
                    res.append(self.unk_idx)  # unk the first char
                    seq = seq[1:]  # delete the first char from seq
                # either cases, reset the states;; and 'seq' should be at least one unit smaller
                prev_node, idx = self.root, 0,
                data_node, data_idx = None, -1

        return [self.vocab[x].name for x in res] if pieces else res

    def decode(self, seq: List[int]) -> List[str]:
        return [self.vocab[i].name for i in seq]

    def encode_all(self, lines: Iterator[str], stringify=True, pieces=False) \
            -> Iterator[Union[List[int], str]]:
        for line in lines:
            line = self.space_tok.join(line.strip().split()) + self.space_tok
            seq = self.encode(line, pieces=pieces)
            if stringify:
                seq = ' '.join(str(x) for x in seq)
            yield seq

    def decode_all(self, lines: Iterator[str], stringify=True) -> Iterator[Union[List[str], str]]:
        for line in lines:
            seq = list(map(int, line.strip().split()))
            seq = self.decode(seq)
            if stringify:
                line = ''.join(str(x) for x in seq)
                seq = line.replace(self.space_tok, ' ').strip()  # restore regular space
            yield seq

    @classmethod
    def read_lines(cls, streams: List[Union[TextIO, Path]]) -> Iterator[str]:
        for i, stream in enumerate(streams):
            stream = stream.open() if isinstance(stream, Path) else stream
            if hasattr(stream, 'name'):
                log.info(f"Reading seqs from {stream.name}")
            for l in stream:
                yield l.strip()
            if stream != streams[i]:
                stream.close()

    @classmethod
    def read_seqs(cls, streams: List[Union[TextIO, Path]], max_seqs=2_000_000, max_len=1024) \
            -> Iterator[Seq]:
        n_seqs = 0
        for line in cls.read_lines(streams):
            seq = [int(x) for x in line.strip().split()]
            yield seq[:max_len]
            n_seqs += 1
            if n_seqs >= max_seqs:
                log.warning(f"Not all data was read; Stopping at {n_seqs} seqs")


#######################################

class WordBPE:
    space = SPACE_TOK[0]
    unk = UNK_TOK[0]

    @classmethod
    def prepare_word(cls, word):
        # mark ending of sequences
        # # TODO: check:  looks like sentence piece adds at the beginning
        # subword-nmt (senrich et al 2016) did </w> at the end;
        # 0.2 of subword-nmt puts last char and </w> together
        return word + cls.space

    @classmethod
    def _make_idxs(cls, voc_idx: Dict[str, int], term_freqs: Dict[str, int]) \
            -> Iterator[Tuple[Seq, int]]:
        """Convert character sequences to char indexed seqs"""
        unk_idx = voc_idx[cls.unk]
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
                       char_coverage=DEF_CHAR_COVERAGE) -> List[Type]:
        assert 0.5 < char_coverage <= 1.0

        term_freqs = {cls.prepare_word(word): freq for word, freq in term_freqs.items()}
        alphabet = coll.defaultdict(int)
        for term, freq in term_freqs.items():
            for ch in term:
                alphabet[ch] += freq
            """TODO: test this behavior; similar to subword-nmt v0.2
            for ch in term[:-2]:  # skip the last two: ending and the whitespace marker
                alphabet[ch] += freq
            alphabet[term[-2:]] += freq  # ending + whitespace marker go together as a single byte
            """
        if char_coverage < 1.0:
            pairs = sorted(alphabet.items(), key=lambda x: x[1], reverse=True)
            tot = sum(v for ch, v in pairs)
            cumm = [[ch, v / tot] for ch, v in pairs]  # normalize by total
            for i in range(1, len(cumm)):  # sum all from left to right
                cumm[i][1] += cumm[i - 1][1]
            includes = {ch: alphabet[ch] for ch, cum in cumm if cum <= char_coverage}
            excludes = {ch: ct for ch, ct in alphabet.items() if ch not in includes}
            log.info(f'char coverage={char_coverage} of {tot}; '
                     f' unked chars count:{sum(excludes.values())} from types:{excludes}')
            alphabet = coll.defaultdict(int, includes)
        else:
            log.info("Character coverage: full")
        vocab = Type.with_reserved_types()  # initial vocab with reserved toks

        [alphabet.pop(v.name) for v in vocab if v.name in alphabet]  # they are already in there
        alphabet = sorted(alphabet.items(), key=lambda x: x[1], reverse=True)  # high freq on top

        vocab += [Type(name, level=Level.char, idx=idx, freq=freq)
                  for idx, (name, freq) in enumerate(alphabet, start=len(vocab))]
        return cls._learn_codes(term_freqs, vocab, min_co_evidence=min_co_evidence,
                                vocab_size=vocab_size)

    @classmethod
    def learn_subwords_from_corpus(cls, corpus: Iterator[str], **kwargs) -> List[Type]:
        """
        :param corpus: line iterator
        :param **kwargs : Refer learn_subwords() args
        """
        term_freq = coll.Counter(word for seq in corpus for word in seq.strip().split())
        return cls.learn_subwords(term_freq, **kwargs)


class BPELearn:
    """
    The core BPE learning algorithm
    fast implementation using linked lists
    Note: this implementation takes relatively more RAM; and that is okay for my usecase
    # TODO: write this in c++ or rust and bind it here
    """

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

        log.info("Going to build corpus stats index; This might take lot of time and memory")
        n_seqs, n_ignored, bar_msg = 0, 0, ''
        with tqdm.tqdm(enumerate(seqs), unit='seqs', dynamic_ncols=True) as data_bar:
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

                if self._is_problematic_seq(seq):
                    n_ignored += 1
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
                bar_msg = f'Seqs: Total={n_seqs} Dropped={n_ignored}; MaxRSS={max_RSS()[1]}'
                data_bar.set_postfix_str(bar_msg)
        log.info(f"Created index; {bar_msg}")
        self.validate_index()

    def _is_problematic_seq(self, seq) -> bool:
        # Wisdom: detect problems ahead and walk away iff
        #  (1) Solving them is not fun. (2) Possible to live without solving them.
        for i in range(2, len(seq)):
            """
            repetitions are bad; we are going to ignore them as of now; for example see below
            case1: seq= x 1 1 y  ; uni={x:1, 1:2, y:1}; bi= {(x,1):1, (1,1)=1, (1, y)=1}
            case2:, seq= x 1 1 1 y; uni={x:1, 1:3, y:1}; bi= {(x,1):1, (1,1)=2, (1, y)=1}
            case3, seq= x 1 1 1 1 y; uni={x:1, 1:4, y:1}; bi= {(x,1):1, (1,1)=3, (1, y)=1}
            => case 1 is okay;  uni[1] -= bi[(1,1)] two times as usual
            => case 2 is bad;   uni[1] -= bi[(1,1)] two times as is a mess up
            => case 3 or longer is really a mess up; total mess up of replacements
            """
            # three or more consecutive same codepoints --> problem!
            if seq[i] == seq[i - 1] == seq[i - 2]:
                return True
        return False

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

            log.info(f"merge :: {new_code} || {a:4}:{uni[a]:5} || {b:4}:{uni[b]:5} || {pair_freq}"
                     f" || {vocab[a].name} {vocab[b].name}")

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
                b_node.delete()  # delete() takes care of linking a → y and a ←y
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


def learn_codes(cmd: str, inp: List[Path], vocab: Path, char_coverage: float = DEF_CHAR_COVERAGE,
                vocab_size_l1: int = 0, min_co_evidence_l1: int = DEF_MIN_CO_EV,
                vocab_size_l2: int = 0, min_co_evidence_l2: int = DEF_MIN_CO_EV, prepared=False):
    if cmd in {'learn', 'learn1'}:
        assert not prepared  # accepts raw input only here
        seqs_raw = BpeCodec.read_lines(inp)
        vocab_model_l1 = WordBPE.learn_subwords_from_corpus(
            seqs_raw, vocab_size=vocab_size_l1, min_co_evidence=min_co_evidence_l1,
            char_coverage=char_coverage)
        BpeCodec.write_vocab(vocab_model_l1, out=vocab)

    if cmd in {'learn', 'learn2'}:
        assert vocab.exists()
        vocab_model_l1 = BpeCodec.read_vocab(vocab)

        if prepared:
            seqs_l1 = BpeCodec.read_seqs(streams=inp)
        else:  # encode using l1 vocab
            codec = BpeCodec(vocab_model_l1)
            seqs_l1 = codec.encode_all(codec.read_lines(inp), stringify=False)
        leaner_l2 = BPELearn(seqs=seqs_l1, vocab=vocab_model_l1)
        vocab_model_l2 = leaner_l2.learn_codes(n_merges=vocab_size_l2,
                                               min_co_evidence=min_co_evidence_l2,
                                               code_level=Level.phrase)
        vocab.rename(vocab.with_suffix(".l1.txt"))  # move old file
        BpeCodec.write_vocab(vocab_model_l2, out=vocab)


def run(cmd: str, inp, vocab: Path, out: TextIO = None, pieces=False, **kwargs):
    assert cmd in ArgParser.cmd_choices
    if cmd.startswith('learn'):
        assert isinstance(inp, list) and isinstance(inp[0], Path)
        learn_codes(cmd, inp, vocab, **kwargs)
    else:
        assert isinstance(inp, TextIO)
        assert isinstance(out, TextIO)
        bpe = BpeCodec(vocab)
        if cmd == 'encode':
            lines = bpe.encode_all(inp, pieces=pieces)
        elif cmd == 'decode':
            lines = bpe.decode_all(inp)
        else:
            raise ValueError(f'{cmd} not supported')
        for i, line in enumerate(lines):
            out.write(line + '\n')


class ArgParser(argparse.ArgumentParser):
    """A class to group all the arg parse stuff.
    You dont need to pay attention here unless you want to edit CLI args spec"""
    cmd_choices = {'learn', 'learn1', 'learn2', 'encode', 'decode'}

    def __init__(self):
        p_args = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                      description=__description__,
                      epilog=__epilog__)
        super().__init__(prog='bpepp', **p_args)
        self.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
        sub_ps = self.add_subparsers(dest='cmd', parser_class=argparse.ArgumentParser)
        sub_ps.required = True

        self.l1_p = sub_ps.add_parser('learn1', help='Learn level1 BPE', **p_args)
        self.l2_p = sub_ps.add_parser('learn2', help='Learn level2 BPE, from level1', **p_args)
        self.l_p = sub_ps.add_parser('learn', help='learn1 and learn2', **p_args)
        self.enc_p = sub_ps.add_parser('encode', help='Encode seqs', **p_args)
        self.dec_p = sub_ps.add_parser('decode', help='Decode seqs', **p_args)

        self.all_ps = [self.l1_p, self.l2_p, self.l_p, self.enc_p, self.dec_p]
        levels = [1, 2, 1, 0, 0]
        for level, sub_p in zip(levels, self.all_ps):
            self.add_iov(sub_p, level)

        for level, parsers in [(1, [self.l1_p, self.l_p]), (2, [self.l2_p, self.l_p])]:
            for parser in parsers:
                self.add_learn_args(parser, level)

    def add_iov(self, parser, level):
        # all of them got vocabulary
        parser.add_argument('-vf', '--vocab-size', '--vocab', dest='vocab',
                            type=Path, help="Vocabulary Path", required=True)

        cmd = parser.prog.split()[-1]
        if cmd in {'encode', 'decode'}:  # encode decode has outputs
            parser.add_argument("inp", nargs="?", default=sys.stdin, type=argparse.FileType('r'),
                                help=f"Input file.")
            parser.add_argument("-o", '--out', default=sys.stdout, type=argparse.FileType('w'),
                                help="Path to output.")
            if cmd == 'encode':
                parser.add_argument("-p", '--pieces', action='store_true',
                                    help='Output word piece string instead of word idx ints')
        else:
            fmt = 'One sentence per line; the tokens should be separated by a regular white space.'
            if cmd == 'learn2':
                fmt = "One sequence per line. The token should be converted to integer ids from " \
                      " learn1/level1 and be separated by regular white spaces."
            # all others which are learn commands take Paths
            parser.add_argument("inp", nargs="+", type=Path, help=f"Input file(s) having {fmt}.")

    def add_learn_args(self, parser, level: int):
        assert level in {1, 2}
        cmd = parser.prog.split()[-1]
        parser.add_argument(f'-mce{level}', f'--min-co-evidence-l{level}', type=int, default=5,
                            help="Minimum frequency of bigrams (co-occurrence evidence) to consider"
                            f" merging of pairs in {level}")
        parser.add_argument(f'-vs{level}', f'--vocab-size-l{level}', type=int, required=True,
                            help=f"vocab size of input sequences for level{level}.")

        if level == 1:
            parser.add_argument('-cc', '--char-coverage', type=float, default=DEF_CHAR_COVERAGE,
                                help='Character coverage, range=[0.5, 1.0]')

        if cmd == 'learn2':
            parser.add_argument('--prepared', action='store_true',
                                help='input is already prepared from level1.')


def main():
    args = ArgParser().parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
