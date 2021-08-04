#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-16

import abc
import collections as coll
import json
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import List, TextIO, Dict, Tuple, Union, Iterator, Optional
import multiprocessing as mp
from tqdm import tqdm
from nlcodec import __version__, log
from nlcodec.dstruct import TrNode
from nlcodec.utils import filter_types_coverage, IO
import os
import sys
import random


N_CPUS = max(1, mp.cpu_count() - 1)
N_CPUS = int(os.environ.get('NLCODEC_THREADS', str(N_CPUS)))

assert N_CPUS >= 1
from nlcodec import DEF_WORD_MIN_FREQ as WORD_MIN_FREQ
from nlcodec import DEF_CHAR_MIN_FREQ as CHAR_MIN_FREQ
from nlcodec import DEF_CHAR_COVERAGE as CHAR_COVERAGE
from nlcodec import DEF_MIN_CO_EV as MIN_CO_EV


class Reseved:
    PAD_TOK = '<pad>', 0
    UNK_TOK = '<unk>', 1  # unk = '⁇'  # U+2047  to make up some OOV characters
    BOS_TOK = '<s>', 2
    EOS_TOK = '</s>', 3
    CLS_TOK = '<cls>', 4
    SPACE_TOK = '▁', 5  # U+2581 same as google/sentencepiece

    PAD_IDX = PAD_TOK[1]
    UNK_IDX = UNK_TOK[1]
    BOS_IDX = BOS_TOK[1]
    EOS_IDX = EOS_TOK[1]
    CLS_IDX = CLS_TOK[1]

    ALL = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK, CLS_TOK, SPACE_TOK]

    @classmethod
    def validate(cls, table: List['Type']):
        for tok, idx in cls.ALL:
            assert tok == table[idx].name, f'Cant find reserved {tok} at index {idx}'

    @classmethod
    def with_reserved_types(cls) -> List['Type']:
        return [Type(tok, level=Level.reserved, idx=idx, freq=-1) for tok, idx in cls.ALL]


class Level:
    reserved = -1
    char = 0
    user = 0  # as of now,  user and char are indistinguishable; 0 means don't look inside
    subword = 1
    word = 2
    phrase = 3
    clasz = 0   # 0 means dont split these tokens


@dataclass(frozen=True)
class Type:  # Type as in word type vs token
    name: str  # name of the type
    level: int  # in [-1, 0, 1, 2, 3]
    idx: int  # idx for modeling integer sequence
    freq: int  # when available; from corpus
    kids: Optional[Tuple['Type', ...]] = None  # in case it was composed by sub pieces

    def format(self, delim='\t') -> str:
        cols = [str(self.idx), self.name, str(self.level), str(self.freq),
                ' '.join(str(k.idx) for k in self.kids) if self.kids else '']
        return delim.join(cols)

    @property
    def is_reserved(self) -> bool:
        return self.level == Level.reserved

    def signature(self) -> str:
        return f'{self.idx}:{self.name}→{"|".join(f"{k.idx}:{k.name}" for k in self.kids or [])}'

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

    def copy(self, **kwargs):
        assert kwargs
        # class is frozen, to update a field, make a copy and set new value
        args = {f.name: kwargs.pop(f.name, getattr(self, f.name)) for f in fields(self)}
        assert not kwargs, f'{kwargs} are unknown'
        return Type(**args)

    @classmethod
    def write_out(cls, table: List['Type'], out: Union[Path, str, TextIO], **meta):

        if isinstance(out, Path) or isinstance(out, str):
            wrtr = open(out, 'w', encoding='utf8', errors='ignore')
        else:
            wrtr = out

        levels = dict(coll.Counter(v.level for v in table))
        max_level = max(levels.keys())
        header = dict(total=len(table), version=__version__, levels=levels, max_level=max_level,
                    created=str(datetime.now()))
        header.update(meta)
        header = json.dumps(header)
        wrtr.write(f"#{header}\n")
        for i, item in enumerate(table):
            assert i == item.idx, f'{item} expected index {i}'
            wrtr.write(item.format() + '\n')
        if wrtr is not out:
            wrtr.close()
        log.info(f"Wrote {len(table)} to {wrtr.name}")

    @classmethod
    def read_vocab(cls, inp: Union[Path, str, TextIO]) -> Tuple[List['Type'], Optional[Dict]]:

        if isinstance(inp, Path) or isinstance(inp, str):
            rdr = open(inp, 'r', encoding='utf8', errors='ignore')
        else:
            rdr = inp

        lines = list(l.strip() for l in rdr)
        meta = None
        if lines[0].startswith("#{"):
            # metadata such as version; not used as of now
            meta = json.loads(lines.pop(0)[1:])

        # noinspection PyTypeChecker
        vocab: List[Type] = [None] * len(lines)
        for i, line in enumerate(lines):
            v = Type.parse(line=line.rstrip('\n'), vocab=vocab)
            assert v.idx == i
            vocab[i] = v
        if rdr is not inp:
            rdr.close()
        log.info(f"read {len(vocab)} types from {rdr.name}")
        return vocab, meta

    def get_permutations(self, name=False) -> List[List[int]]:
        """
        gets all possible permutations of kids that could compose thise
        :param name: get name of pieces instead of ids. default=False
        :return: List of permutations
        """
        perms = [[self.name if name else self.idx]]
        if self.kids:
            left, right = self.kids
            for left_kid in left.get_permutations(name=name):
                for right_kid in right.get_permutations(name=name):
                    perms.append(left_kid + right_kid)
        return perms

    def get_stochastic_split(self, name=False, split_ratio=0.1):
        if self.kids and random.random() < split_ratio:
            left, right = self.kids
            return left.get_stochastic_split(name=name, split_ratio=split_ratio) \
                   + right.get_stochastic_split(name=name, split_ratio=split_ratio)
        else:
            return [self.name if name else self.idx]


class EncoderScheme(abc.ABC):

    name = ''

    def __init__(self, table: List[Type], has_reserved=True, invertible=True):
        """

        :param table: list of `Type`s
        :param has_reserved: validate that reserved types are found
        :param invertible: validate that the idx->str and str->idx are invertible
        """
        if has_reserved:
            Reseved.validate(table)
            self.unk_idx = Reseved.UNK_IDX

        self.vocab_size = len(table)
        self.table = table
        self.idx_to_str = [t.name for t in table]
        if invertible:
            self.str_to_idx = {tok: idx for idx, tok in enumerate(self.idx_to_str)}
            assert len(self.idx_to_str) == len(self.str_to_idx)
        else:
            # keep the first occurrence TODO: maybe keep both and do random; str_to_idx be multiset
            self.str_to_idx = {}
            for idx, typ in enumerate(table):
                if typ.name in self.str_to_idx:
                    typ2 = table[self.str_to_idx[typ.name]]
                    log.debug(f"skip:: {typ.signature()}; it conflicts with {typ2.signature()}")
                else:
                    self.str_to_idx[typ.name] = idx
        self.invertible = invertible

    def __len__(self):
        return self.vocab_size


    @abc.abstractmethod
    def encode_str(cls, line: str) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def decode_str(cls, seq: List[str]) -> str:
        raise NotImplementedError()

    def encode(self, line: str) -> List[int]:
        pieces = self.encode_str(line)
        return [self.str_to_idx.get(piece, self.unk_idx) for piece in pieces]

    def decode(self, seq: List[int]) -> str:
        pieces = [self.idx_to_str[idx] for idx in seq]
        return self.decode_str(pieces)

    def encode_parallel(self, seqs: Iterator[str], n_cpus=N_CPUS) -> Iterator[List[int]]:
        return self.parallel_map(self.encode, seqs, n_cpus=n_cpus)

    @classmethod
    def parallel_map(cls, mapper, collection, n_cpus=N_CPUS, name='', chunksize=1000):
        assert n_cpus > 1, f'at least 2 CPUs needed. chunksize={chunksize}'
        log.info(f"Going to use {n_cpus} parallel processes {name}")
        with mp.Pool(processes=n_cpus) as pool:
            yield from pool.imap(mapper, collection, chunksize=chunksize)

    @classmethod
    @abc.abstractmethod
    def learn(cls, data: Iterator[str], **kwargs) -> List[Type]:
        raise NotImplementedError()

    @classmethod
    def get_init_vocab(cls, term_freqs, coverage: float = 0, line_count=None,
                       min_freq=WORD_MIN_FREQ,
                       vocab_size=-1):
        vocab = Reseved.with_reserved_types()
        res_stats = {r_type.name: term_freqs.pop(r_type.name) for r_type in vocab if
                     r_type.name in term_freqs}
        if res_stats:
            log.warning(f"Found reserved types in corpus: {res_stats}")
        # Order of trimming techs: 1. coverage, 2. min freqs, 3. size cut off
        unk_count = 0
        if coverage:
            assert 0 < coverage <= 1
            term_freqs, coverage_unk_count = filter_types_coverage(term_freqs, coverage=coverage)
            unk_count += coverage_unk_count
        term_freqs = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)
        if min_freq and min_freq > 1:
            log.info(f"Excluding terms with freq < {min_freq}; |freq >= 1|: {len(term_freqs):,}")
            unk_count += sum(f for t, f in term_freqs if f < min_freq)
            term_freqs = [(t, f) for t, f in term_freqs if f >= min_freq]
            log.info(f"|freq >= {min_freq}| : {len(term_freqs):,}")

        if vocab_size > 0 and len(vocab) + len(term_freqs) > vocab_size:
            log.info(f"Truncating vocab at size={vocab_size}")
            unk_count += sum(f for t, f in term_freqs[vocab_size - len(vocab):])
            term_freqs = term_freqs[:vocab_size - len(vocab)]

        # update reserved types with corpus freqs
        for idx, t in enumerate(vocab):
            freq = 0
            if t.name in res_stats:
                freq = res_stats.pop(t.name)
            if idx == Reseved.UNK_IDX:
                freq += unk_count
            if idx in {Reseved.BOS_IDX, Reseved.EOS_IDX, Reseved.CLS_IDX} and line_count:
                freq += line_count
            if freq:
                log.warning(f"Update frequency for reserved type {t} with {freq}")
                vocab[idx] = t.copy(freq=freq)
        vocab += [Type(name=name, idx=idx, freq=freq, level=cls.level)
                  for idx, (name, freq) in enumerate(term_freqs, start=len(vocab))]
        log.info(f"Total {cls} vocab size {len(vocab):,}")
        return vocab

    def shrink_vocab(self, files: List[Path], min_freq: int, save_at: Optional[Path] = None) -> List[int]:
        """
        :param files:
        :param min_freq:
        :param save_at:
        :return:
        """
        """"
- Accept a list of files
- compute term frequencies
- Eliminate types with zero counts
- Preserve reserved types even if they have zero counts
- Save the resulting model at a given file path
- Return index mapping between old and new, so we can go back to model and shrink embedding tables
        """
        from tqdm import tqdm
        freqs = coll.Counter()
        for file in files:
            log.info(f'Computing term frequencies from {file}')
            with IO.reader(file) as lines:
                for line in tqdm(lines):
                    freqs.update(self.encode(line))
        assert len(self.table) > max(freqs.keys())
        removals = [False] * len(self.table)
        for idx, typ in enumerate(self.table):
            if typ.level == Level.reserved:
                continue # i.e. don't remove
            removals[idx] = freqs[idx] < min_freq  # remove if min_freq threshold not met

        # now make sure to preserve all the sub pieces leading to the pieces that retain
        for idx in range(len(self.table) - 1, -1, -1):
            combo = self.table[idx]
            assert combo.idx == idx
            if not removals[combo.idx] and combo.kids: #
                for piece in combo.kids:
                    if removals[piece.idx]:
                        removals[piece.idx] = False   # dont remove this piece,

        mapping = []
        for idx, (is_remove, typ) in enumerate(zip(removals, self.table)):
            assert idx == typ.idx
            if is_remove:
                continue
            mapping.append(idx)

        log.info(f"Shrinking vocab tables: {len(self.table)} --> {len(mapping)} ")
        rev_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(mapping)}
        old_table = self.table
        new_table = []
        for new_idx, old_idx in enumerate(mapping):
            assert len(new_table) == new_idx
            old_t = old_table[old_idx]
            new_kids = [new_table[rev_mapping[k.idx]] for k in old_t.kids] if old_t.kids else None
            new_t = old_t.copy(idx=new_idx, kids=new_kids)
            new_table.append(new_t)
        if save_at:
            Type.write_out(new_table, out=save_at)
        return mapping

    def save(self, out):
        return Type.write_out(table=self.table, out=out, scheme=self.name)


class WordScheme(EncoderScheme):
    level = Level.word
    name = "word"

    @classmethod
    def encode_str(cls, line: str) -> List[str]:
        return line.split()

    @classmethod
    def decode_str(cls, seq: List[str]) -> str:
        return " ".join(seq)

    @classmethod
    def term_frequencies(cls, data: Iterator[str]) -> Tuple[Dict[str, int], int]:
        stats = coll.Counter()
        line_count = 0
        for line in tqdm(data, mininterval=1):
            stats.update(cls.encode_str(line.strip()))
            line_count += 1
        log.info(f"Found {len(stats):,} types and {sum(stats.values()):,} tokens")
        return stats, line_count

    @classmethod
    def read_term_freqs(cls, data: Iterator[str], delim='\t') -> Tuple[Dict[str, int], int]:
        stats = {}
        line_count = -1
        for idx, line in enumerate(data):
            line = line.rstrip('\n')
            if idx == 0 and line.startswith("#") and len(line.split(delim)) != 2:
                try:
                    import json
                    meta = json.loads(line[1:])  # skip # at index 0
                    line_count = meta.get('line_count', line_count)
                except:
                    pass
                continue
            term, freq = line.split("\t")
            stats[term.strip()] = int(freq)
        return stats, line_count

    @classmethod
    def learn(cls, data: Iterator[str], vocab_size: int = 0, min_freq: int = WORD_MIN_FREQ,
              coverage: float = 0, term_freqs=False, **kwargs) -> List[Type]:
        """
        :param data: input sentences
        :param vocab_size: max vocabulary size.
        :param min_freq: min frequency for inclusion in vocabulary. Excludes types with lower freq
        :param coverage: Character coverage
        :param term_freqs: is data the term_freqs ?
        :param kwargs: place holder for any extra args
        :return:
        """
        assert not kwargs, f'{kwargs} args are not allowed/understood'
        if term_freqs: # input is term_freqs
            log.info("Restoring term frequencies from input")
            stats, line_count = cls.read_term_freqs(data=data)
        else: # compute term freqs
            log.info("Computing term frequencies from raw data")
            stats, line_count = cls.term_frequencies(data=data)

        return cls.get_init_vocab(stats, coverage, line_count, min_freq, vocab_size)


class CharScheme(WordScheme):
    space_char = '▁'  # U+2581 same as google/sentencepiece
    level = Level.char
    name = "char"

    @classmethod
    def encode_str(cls, line: str) -> List[str]:
        return list(cls.space_char.join(line.split()))

    @classmethod
    def decode_str(cls, seq: List[str]) -> str:
        return ''.join(seq).replace(cls.space_char, ' ')

    @classmethod
    def learn(cls, data: Iterator[str], vocab_size: int = 0, min_freq: int = CHAR_MIN_FREQ,
              coverage: float = CHAR_COVERAGE, **kwargs) -> List[Type]:
        # learn() is same as parent class: WordScheme
        return super().learn(data, vocab_size, min_freq=min_freq, coverage=coverage, **kwargs)


class BPEScheme(CharScheme):
    level = Level.subword
    name = "bpe"

    def __init__(self, table: List[Type]):
        super().__init__(table=table, invertible=False)
        self.root = self.make_vocab_prefix_trie(self.table)
        log.info(f"Vocab size={len(self)}; trie root has nodes={self.root.size}"
                 f" but data_nodes={self.root.data_node_count}")
        assert self.unk_idx

    @classmethod
    def make_vocab_prefix_trie(cls, vocab: List[Type]):
        root: TrNode = TrNode[str, Type](idx='')
        for typ in vocab:
            node = root.get_node(idxs=typ.name, create_missing=True)
            node.name = typ.name
            node.data = typ
        assert not root.has_data  # root node is not data node
        return root


    def encode(self, line: str, split_ratio: float = 0.) -> List[int]:
        pieces = self.encode_str(line, split_ratio=split_ratio)
        return [self.str_to_idx.get(piece, self.unk_idx) for piece in pieces]

    def encode_str(self, line: str, split_ratio: float = 0.) -> List[str]:
        seq = self.space_char.join(line.strip().split()) + self.space_char
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
                # either cases, reset the states; and 'seq' should be at least one unit smaller
                prev_node, idx = self.root, 0,
                data_node, data_idx = None, -1

        if split_ratio > 0:
            res = self.stochastic_split(res, split_ratio=split_ratio, name=False)

        return [self.table[idx].name for idx in res]

    def decode_str(self, seq: List[str]) -> str:
        return ''.join(seq).replace(self.space_char, ' ').strip()

    @classmethod
    def learn(cls, data: Iterator[str], vocab_size: int = 0, min_freq=WORD_MIN_FREQ,
              coverage=CHAR_COVERAGE, min_co_evidence=MIN_CO_EV, term_freqs=False, **kwargs) -> List[Type]:
        if min_co_evidence > 1 and vocab_size <= 1:
            log.info(f'Using Min-co-evidence={min_co_evidence} as stop condition. '
                     f'Treating vocab_size as maximum size')
            vocab_size = sys.maxsize
        assert vocab_size > 0
        assert not kwargs, f'{kwargs} args are not allowed/understood'
        if term_freqs:
            log.info("Reading term freqs from input")
            tfs, line_count = WordScheme.read_term_freqs(data)
        else:
            log.info("Computing term freqs from input")
            tfs, line_count = WordScheme.term_frequencies(data)

        def init_vocab_factory(char_types):
            return CharScheme.get_init_vocab(char_types, line_count=line_count,
                                             coverage=coverage, min_freq=1)

        from .bpe import BPELearn
        vocab = BPELearn.learn_subwords(term_freqs=tfs, vocab_size=vocab_size,
                                        init_vocab_factory=init_vocab_factory,
                                        min_co_evidence=min_co_evidence)
        return vocab

    def stochastic_split(self, seq, split_ratio, name=False):
        res = []
        for idx in seq:
            res += self.table[idx].get_stochastic_split(name=name, split_ratio=split_ratio)
        return res

class ClassScheme(WordScheme):
    """Scheme to be used for mapping labels or classes"""
    level = Level.clasz
    name = "class"
    delim = ","
    unk_idx = -1    # No unks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, has_reserved=False, **kwargs)


    @classmethod
    def encode_str(cls, line: str) -> List[str]:
        return [line.strip()]

    def decode_str(self, seq: List[str]) -> str:
        return self.delim.join(seq)

    @classmethod
    def get_init_vocab(cls, term_freqs, *args, **kwargs):
        vocab = []
        # sort alphabetically
        term_freqs = sorted(term_freqs.items(), key=lambda x: x[0], reverse=False)

        vocab += [Type(name=name, idx=idx, freq=freq, level=cls.level)
                  for idx, (name, freq) in enumerate(term_freqs, start=len(vocab))]
        log.info(f"Total {cls} vocab size {len(vocab):,}")
        return vocab


#########################
REGISTRY = {
    'char': CharScheme,
    'word': WordScheme,
    'bpe': BPEScheme,
    'subword': BPEScheme,
    'class': ClassScheme
}


def learn_vocab(inp, level, model, vocab_size, min_freq=1, term_freqs=False,
                char_coverage=CHAR_COVERAGE, min_co_ev=MIN_CO_EV) -> List[Type]:
    if not min_freq or min_freq < 1:
        min_freq = WORD_MIN_FREQ if level == 'word' else CHAR_MIN_FREQ
        log.info(f"level={level} => default min_freq={min_freq}")
    else:
        log.info(f"level={level} => user given min_freq={min_freq}")
    log.info(f"Learn Vocab for level={level} and store at {model}")
    if isinstance(inp, (list, tuple)):
        log.info(f"data ={inp[:10]} ... + ({len(inp) - 10} more items)")
    else:
        log.info(f"data ={inp}")
    Scheme = REGISTRY[level]
    args = {}
    if level != 'word':
        args['coverage'] = char_coverage  # no char_coverage for word
    if level == 'bpe':
        args['min_co_evidence'] = min_co_ev
    table = Scheme.learn(inp, vocab_size=vocab_size, min_freq=min_freq, term_freqs=term_freqs,
                         **args)
    if model:
        Scheme(table).save(out=model)
    return table


def load_scheme(path: Union[str, Path, TextIO]) -> EncoderScheme:
    types, meta = Type.read_vocab(path)
    assert meta
    if 'scheme' in meta:
        Scheme = REGISTRY[meta['scheme']]
    else:
        # backward compatibility;
        max_level = meta['max_level']
        levels = [CharScheme, BPEScheme, WordScheme, ClassScheme]
        assert max_level < len(levels)
        Scheme = levels[max_level]
    return Scheme(table=types)


def encode(inp: Iterator[str], scheme: EncoderScheme, indices=False) \
        -> Iterator[Union[List[str], List[int]]]:
    for line in inp:
        if indices:
            yield scheme.encode(line)
        else:
            yield scheme.encode_str(line)


def decode(inp: Iterator[str], scheme: EncoderScheme, indices=False) -> Iterator[str]:
    for line in inp:
        seq = line.split()
        if indices:
            seq = [int(x) for x in seq]
            line = scheme.decode(seq)
        else:
            line = scheme.decode_str(seq)
        yield line
