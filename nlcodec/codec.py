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
from typing import List, TextIO, Dict, Tuple, Union, Iterator, Optional, Set
import multiprocessing as mp
from tqdm import tqdm
from nlcodec import __version__, log
from nlcodec.dstruct import TrNode
from nlcodec.utils import filter_types_coverage, IO
from nlcodec.pmi import PMIFuncs
import os
import sys
import random
import functools as fn


N_CPUS = max(1, mp.cpu_count() - 1)
N_CPUS = int(os.environ.get('NLCODEC_THREADS', str(N_CPUS)))

assert N_CPUS >= 1
from nlcodec import DEF_WORD_MIN_FREQ as WORD_MIN_FREQ
from nlcodec import DEF_CHAR_MIN_FREQ as CHAR_MIN_FREQ
from nlcodec import DEF_MWE_MIN_FREQ as MWE_MIN_FREQ
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
        # Disabling to work with the scripts ( prep scripts )
        # assert not kwargs, f'{kwargs} args are not allowed/understood'
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


class NgramScheme(BPEScheme):

    def __init__(self, table:List['Type']):
        log.info('Loading Ngram Scheme ...')
        super().__init__(table=table)

    def decode(self, seq:List[int]) -> str:
        pieces = [self.table[x].name for x in seq]
        return self.decode_str(pieces)

    def decode_str(self, seq:List[str]) -> str:
        return ''.join(seq).replace(self.space_char, ' ').strip()

    @classmethod
    def get_bpe_words(cls, bpe, words_set):
        available_words = set()
        for token in bpe:
            name = token.name
            if name.endswith(Reseved.SPACE_TOK[0]):
                if name[:-1] in words_set:
                    available_words.add(name)
        return available_words
        
    @classmethod
    def ngram_frequencies(cls, data: Iterator[str], 
                            ngram:int) -> Dict[str,int]:
        ngram_freqs = coll.Counter()
        for line in tqdm(data, mininterval=1):
            words = WordScheme.encode_str(line)
            ngrams = [ cls.space_char.join([*words[i:i+ngram], '']) 
                        for i in range((len(words)+1)-ngram)]
            ngram_freqs.update(ngrams)
        return ngram_freqs

    @classmethod
    def sorted_ngrams(cls, ngram_freqs:Dict[str,int], 
        term_freqs:Dict[str, int], nlines:int, metric:str, bigram_freqs=None, 
        min_freq:int=MWE_MIN_FREQ) -> List[Tuple['Type', Union[int,float]]]:
        
        nterms = sum(term_freqs.values())
        ngrams_list = []
        for name, freq in ngram_freqs.items():
            if freq >= min_freq:
                words = name.split(cls.space_char)[:-1]
                word_freqs = [term_freqs[word] for word in words]
                ngrams_list.append(Type(name, freq=freq, idx=0, 
                                    level=1, kids=word_freqs))

        if metric == 'freq':
            sorted_list = [(x,x.freq) for x in ngrams_list]
        else:
            sorted_list = PMIFuncs.get_pmis(ngrams_list, nterms, nlines, 
                                            bigram_freqs=bigram_freqs,
                                            pmi_variant=metric)
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        return sorted_list
            
    @classmethod
    def filtered_ngrams(cls, 
            ngrams_list:List[Tuple['Type',Union[int,float]]],
            bpes:List['Type'],
            words_set:Set[str]) -> List[Tuple['Type', Union[int,float]]]:
        
        rev_idx = {t.name:t.idx for t in bpes}
        filtered = []
        for pair in ngrams_list:
            tok, val = pair
            parts = tok.name.replace(cls.space_char, f'{cls.space_char} ').split()
            not_word = [ part not in words_set for part in parts]
            if not any(not_word):
                kids = [bpes[rev_idx[x]] for x in parts]
                tok = Type(tok.name, 4, tok.idx, tok.freq, kids=kids)
                filtered.append((tok, val))
        return filtered

    @classmethod
    def get_ngrams_lists(cls, data:Iterator[str], ngrams:List[int]=None, 
                        sorter_func:str='freq', min_freq:int=MWE_MIN_FREQ, 
                        vocab_size:int=0, bpe_vocab:List['Type']=None):
        assert ngrams is not None
        assert vocab_size != 0 or bpe_vocab is not None
        assert sorter_func == 'freq' or sorter_func in PMIFuncs.ngram_variants

        if bpe_vocab is None:
            bpe_vocab = BPEScheme.learn(data, vocab_size)

        term_freqs, nlines = WordScheme.term_frequencies(data)
        all_words = set(term_freqs.keys())
        bpe_words = cls.get_bpe_words(bpe_vocab, all_words)

        ngrams_lists = {}
        
        bigram_freqs = cls.ngram_frequencies(data, 2)
        for ng in ngrams:
            ngram_freqs = cls.ngram_frequencies(data, ng)
            sorted_ngrams = cls.sorted_ngrams(ngram_freqs, term_freqs,
                                    nlines, sorter_func, 
                                    bigram_freqs=bigram_freqs,
                                    min_freq=min_freq)
            ngrams_lists[ng] = cls.filtered_ngrams(sorted_ngrams, 
                                                bpe_vocab, bpe_words)
        return ngrams_lists, bpe_vocab

    @classmethod
    def merge_lists(cls, base, lists, vocab_size, grams, toks_list):
        unk_idx = Reseved.UNK_IDX
        base_len = vocab_size - sum(toks_list)

        vocab = base[:base_len]
        for gram, toks in zip(grams, toks_list):
            trimmed_list =lists[gram][:toks]
            for pair in trimmed_list:
                tok = pair[0]
                # Doubt : How to add the ngrams ??
                # 1. Use a global list of ngrams irrespective of the words in the vocabs
                # 2. Consider words in the base vocab (all) [current]
                # 3. Consider words in the base vocab (non-replaced)
                # if any([t.idx >= base_len for t in tok.kids]):
                kids = [t if t.idx < base_len else base[unk_idx] for t in tok.kids]
                vocab.append(Type(tok.name, tok.level, len(vocab), 
                                tok.freq, kids))
        return vocab

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, ngrams:List[int]=None, 
            max_ngrams:int=0, ngram_sorter:str='freq', toks_list:List[int]=[],
            min_freq:int=MWE_MIN_FREQ, **kwargs) -> List['Type']:

        assert ngrams is not None
        assert len(toks_list) == len(ngrams) or max_ngrams > 0
        
        base = BPEScheme.learn(data, vocab_size)
        ngrams_lists, _ = cls.get_ngrams_lists(data, ngrams, ngram_sorter,
                                            min_freq, bpe_vocab=base)

        # Currently equal number of ngrams from each list are included
        # or else provided by the user themselves
        if len(toks_list) == 0:
            toks_list = [max_ngrams // len(ngrams)] * len(ngrams)
        assert vocab_size > sum(toks_list)
        
        return cls.merge_lists(base, ngrams_lists, vocab_size, 
                                ngrams, toks_list)


class SkipScheme(BPEScheme):

    PLACE_TOK = '▂', 6 # U+2582 ??
    SKIP_TOK = '<skp>', 7
    TOKS = [PLACE_TOK, SKIP_TOK]
    
    skip_char = PLACE_TOK[0]
    skip_tok = SKIP_TOK[0]

    count = 1

    hash_prime = 9973 # prime number to hash the list
    
    def __init__(self, table:List['Type']):
        log.info('Loading Skip Scheme ...')
        super().__init__(table=table)
        self.root = self.make_vocab_prefix_trie(self.table)
        assert self.unk_idx

    def encode(self, line: str, split_ratio: float = 0.) -> List[int]:
        pieces = self.encode_str(line, split_ratio=split_ratio)
        return [self.str_to_idx.get(piece, self.unk_idx) for piece in pieces]

    def encode_str(self, line:str, split_ratio=None) -> List[str]:
        seq = self.space_char.join(line.strip().split()) + self.space_char
        res: List[int] = []

        def _set_default():
            return None, False, True

        back_pairs = []
        data_node, data_idx = None, -1
        prev_node, idx = self.root, 0
        tokens, is_skip, check_skip = _set_default()

        # Thought of a little tweak to make it work for a_b_c tokens.
        # Setting global tokens to some value. And naming while token to 
        # something else.

        while seq and idx < len(seq)+1:

            if prev_node.has_data:
                data_node, data_idx = prev_node, idx

            if self.skip_char in prev_node.kids and check_skip and idx != 0:

                tokens, ahead_pair = self.get_skips(seq, idx, prev_node)
                if tokens is not None:
                    back_pair = (prev_node, idx)
                    back_pairs.append(back_pair)
                    prev_node, idx = ahead_pair
                    is_skip = True
                    check_skip = False
                else:
                    check_skip = False
            else:
                if idx < len(seq) and seq[idx] in prev_node.kids:
                    prev_node = prev_node.kids[seq[idx]]
                    idx += 1
                else:
                    if data_node:
                        res.append(data_node.data.idx)
                        seq = seq[data_idx:]
                        if is_skip:
                            res.extend(tokens)
                            is_skip = False
                    else:
                        res.append(self.unk_idx)
                        seq = seq[1:]

                    back_pairs = []
                    prev_node, idx = self.root, 0
                    data_node, data_idx = None, -1
                    tokens, is_skip, check_skip = _set_default()
        
        return [self.table[idx].name for idx in res]
                    
    def check_skippable(self, seq, pos, curr_node):
        tseq = seq[pos:]
        ix = 0
        while ix < len(tseq):
            if tseq[ix] in curr_node.kids:
                curr_node = curr_node.kids[tseq[ix]]
                ix += 1
                if curr_node.has_data:
                    return True
            else:
                return False        
        return False
   
    def get_skips(self, seq, pos, node):
        next_idxs = []
        next_tokens = []
        prev_node, idx = self.root, pos
        while idx < len(seq):
            if seq[idx] in prev_node.kids:
                prev_node = prev_node.kids[seq[idx]]
                idx += 1
                if prev_node.has_data:
                    next_idxs.append(idx)
                    next_tokens.append(prev_node)
            else:
                break

        next_node = node.kids[self.skip_char]
        next_idxs.reverse()
        next_tokens.reverse()

        if self.skip_char in next_node.kids:
            for token, next_idx in zip(next_tokens, next_idxs):
                tokens, ahead_pair = self.get_skips(seq, next_idx, next_node)
                if tokens is not None:
                    tokens.insert(0, token.data.idx)
                    return tokens, ahead_pair

        for token, next_idx in zip(next_tokens, next_idxs):
            if self.check_skippable(seq, next_idx, next_node):
                return [token.data.idx, self.SKIP_TOK[1]], (next_node, next_idx)

        return None, None

    # This implementation only works with 1-skip tokens.
    def decode_str(self, seq: List[str]) -> str:
        decoded_seq = []
        to_add = set()
        tok_to_add = dict()

        idx = 0
        pos = 0
        while idx < len(seq):
            if pos in to_add:
                decoded_seq.append(tok_to_add[pos])
                to_add.remove(pos)
                pos += 1
                continue

            if self.skip_char not in seq[idx]:
                decoded_seq.append(seq[idx])
            else:
                toks = seq[idx].split(self.skip_char)
                decoded_seq.append(toks[0])
                to_add.add(pos+2)
                tok_to_add[pos+2]=toks[1]

            idx += 1
            pos += 1

        if len(to_add) != 0:
            decoded_seq.append(self.skip_tok)
            decoded_seq.append(tok_to_add[pos+1])

        return ''.join(decoded_seq).replace(self.space_char, ' ')

    # @classmethod
    # def decode_str(cls, seq:List[str]) -> str:
        
    #     ordered_seq = [None]*len(seq)
        
    #     decoded_seq = []
    #     to_add = set()
    #     tok_to_add = dict()

    #     pos = 0
    #     idx = 0

    #     while idx < len(seq):
    #         if pos in to_add:
    #             decoded_seq.append(tok_to_add[pos])
    #             to_add.remove(pos)
    #             continue



    #     skipped_pos = []
    #     try:
    #         for tok in seq:
    #             if tok == cls.skip_tok:
    #                 continue
    #             nskips = coll.Counter(tok).get(cls.skip_char,0)
    #             if nskips:
    #                 xtok = tok.replace(cls.skip_char, f'{cls.skip_char} ')
    #                 xtok = xtok.replace(cls.space_char, f'{cls.space_char} ')
    #                 parts = xtok.split()
    #                 for ix, part in enumerate(parts):
    #                     if part != cls.skip_char:
    #                         ordered_seq[pos+ix] = part
    #                     else:
    #                         skipped_pos.append(pos+ix)
    #                 pos += len(parts)
    #             else:
    #                 if len(skipped_pos) != 0:
    #                     cpos = skipped_pos[0]
    #                     skipped_pos = skipped_pos[1:]
    #                     ordered_seq[cpos] = tok
    #                 else:
    #                     ordered_seq[pos] = tok
    #                     pos += 1
    #     except Exception as e:
    #         print(seq)
    #         raise(e)
    #     return ''.join(ordered_seq).replace(cls.space_char, ' ')

    @classmethod
    def skipgram_frequencies(cls, data:Iterator[str], 
                    sgram:Tuple[int,int]) -> Dict[str, Dict[str,int]]:
        sgram_freqs = dict()
        _, skip = sgram
        # skip_str = cls.space_char.join([cls.skip_char]*skip) + cls.space_char
        skip_str = cls.skip_char * skip
        for line in tqdm(data, mininterval=1):
            words = WordScheme.encode_str(line)
            nwords = len(words)
            if nwords > skip+1:
                words = [ f'{word}{cls.space_char}' for word in words ]
                for i in range(nwords-(skip+1)):
                    name = f'{words[i]}{skip_str}{words[i+skip+1]}'
                    if name not in sgram_freqs.keys():
                        sgram_freqs[name] = coll.Counter()
                    sgram_freqs[name].update([''.join(words[i+1:i+skip+1])])
        return sgram_freqs

    @classmethod
    def sorted_sgrams(cls, sgram_freqs:Dict[str, Dict[str,int]],
            term_freqs:Dict[str,int], nlines:int, metric:str,
            min_freq:int=0) -> List[Tuple['Type', Union[int,float], Tuple[int,float]]]:
        nterms = sum(term_freqs.values())
        sgrams_list = []
        sgrams_stats = {}
        for name, instances in sgram_freqs.items():
            freq = sum(instances.values())
            if freq < min_freq:
                continue
            
            exname = name.replace(cls.space_char, f'{cls.space_char} ')
            exname = exname.replace(cls.skip_char, f'{cls.skip_char} ')
            words = exname.split()
            word_freqs = [0 if cls.skip_char in word else term_freqs[word[:-1]]
                            for word in words]
            ninstances = len(instances.keys())
            max_prob =  max([val/freq for val in instances.values()])
            sgrams_list.append(Type(name, freq=freq, idx=0,
                                    level=1, kids=word_freqs))
            sgrams_stats[name] = (ninstances, max_prob)

        if metric == 'freq':
            sorted_list = [(x,x.freq) for x in sgrams_list]
        else:
            sorted_list = PMIFuncs.get_pmis(sgrams_list, 
                                            nterms, nlines,  
                                            pmi_variant=metric)
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        sorted_list = [ (tok, val, sgrams_stats[tok.name]) for tok, val in sorted_list]
        return sorted_list

    @classmethod
    def filtered_sgrams(cls, sgrams_list:List[Tuple['Type', Union[int,float], 
            Tuple[int,float]]], bpes:List['Type'], all_words:Set[str], max_instance_prob:float=1.0, 
            min_instances:int=0) -> List[Tuple['Type', Union[int,float], Tuple[int,float]]]:
        rev_idx = {t.name:t.idx for t in bpes}
        all_words.add(cls.skip_char)
        filtered = []
        for trp in sgrams_list:
            tok, val, stats = trp
            ninstances, max_prob = stats
            if ninstances < min_instances or max_prob > max_instance_prob:
                continue
            xname = tok.name.replace(cls.skip_char, f'{cls.skip_char} ')
            parts = xname.replace(cls.space_char, f'{cls.space_char} ').split()
            not_word = [ part not in all_words 
                                for part in parts]
            if not any(not_word):
                kids = [bpes[rev_idx[x]] for x in parts]
                tok = Type(tok.name, 5, tok.idx, 
                                tok.freq, kids)
                filtered.append((tok, val, stats))
        return filtered

    @classmethod
    def get_sgrams_lists(cls, data:Iterator[str], sgrams:List[Tuple[int,int]]=None,
                        sorter_func:str='freq', min_freq:int=MWE_MIN_FREQ,
                        min_instances:int=15, max_instance_prob:float=0.1,
                        vocab_size:int=0, bpe_vocab:List['Type']=None):
        assert sgrams is not None
        assert sorter_func == 'freq' or sorter_func in PMIFuncs.sgram_variants
        assert vocab_size != 0 or bpe_vocab is not None

        print('Found term freqs and nlines')
        term_freqs, nlines = WordScheme.term_frequencies(data)
        all_words = set(term_freqs.keys())

        sgrams_list = {}

        if bpe_vocab is None:
            
            def init_vocab_factory(char_types):
                tvcb = CharScheme.get_init_vocab(char_types, line_count=nlines,
                                                coverage=CHAR_COVERAGE, 
                                                min_freq=1)
                vocab = Reseved.with_reserved_types()
                for tok in cls.TOKS:
                    name, idx = tok
                    vocab.append(Type(name, level=-1, idx=idx, freq=0))
                for tok in tvcb:
                    if tok.level < 0:
                        continue
                    vocab.append(Type(tok.name, level=tok.level, 
                                    idx=len(vocab), freq=tok.freq,
                                    kids=tok.kids))
                return vocab

            from nlcodec.bpe import BPELearn
            bpe_vocab = BPELearn.learn_subwords(term_freqs=term_freqs, 
                                            vocab_size=vocab_size,
                                            init_vocab_factory=init_vocab_factory,
                                            min_co_evidence=MIN_CO_EV)

        bpe_words = NgramScheme.get_bpe_words(bpe_vocab, all_words)
        print('Total Words : ', len(bpe_words))

        print('Making skipgrams')
        for sg in sgrams:
            print(f'> Preparing skipgrams {str(sg)}')
            sgram_freqs = cls.skipgram_frequencies(data, sg)
            sorted_sgrams = cls.sorted_sgrams(sgram_freqs, term_freqs,
                                            nlines, sorter_func, min_freq)
            del(sgram_freqs)
            hash = (sg[0]*cls.hash_prime) + sg[1]
            sgrams_list[hash] = cls.filtered_sgrams(sorted_sgrams, bpe_vocab,
                                                    bpe_words,
                                                    max_instance_prob, 
                                                    min_instances)
        return sgrams_list, bpe_vocab

    @classmethod
    def merge_lists(cls, base, lists, vocab_size, grams, toks_list):
        return NgramScheme.merge_lists(base, lists, vocab_size, grams, toks_list)

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, 
                sgrams:List[Tuple[int,int]]=None, max_sgrams:int=0, 
                skipgram_sorter:str='freq', toks_list:List[int]=[],
                min_freq:int=MWE_MIN_FREQ, min_instances:int=15, 
                max_instance_prob:float=0.1, **kwargs) -> List['Type']:
        assert sgrams is not None
        assert max_sgrams > 0 or len(toks_list) == len(sgrams)

        ## Currently support for n-skip-2 grams only. Need to discuss
        #  about this with AVT

        sgrams_lists, base = cls.get_sgrams_lists(data, sgrams, skipgram_sorter,
                                            min_freq, min_instances,
                                            max_instance_prob,
                                            vocab_size=vocab_size)

        if len(toks_list) == 0:
            toks_list = [max_sgrams // len(sgrams)] * len(sgrams)
        assert vocab_size > sum(toks_list)

        hashed_sgrams = list(map(lambda x: (x[0]*cls.hash_prime) + x[1], sgrams))
        return cls.merge_lists(base, sgrams_lists, vocab_size,
                                hashed_sgrams, toks_list)


class MWEScheme(SkipScheme):
    
    get_ngrams_lists = NgramScheme.get_ngrams_lists

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, 
            mwes:List[Union[int,Tuple[int,int]]]=None, 
            ngram_sorter:str='freq', skipgram_sorter:str='freq',
            toks_list:List[int]=[], max_mwes:int=0,
            min_freq:int=MWE_MIN_FREQ, min_instances:int=15,
            max_instance_prob:float=0.1, **kwargs) -> List['Type']:
        assert mwes is not None
        assert max_mwes > 0 or len(toks_list) == len(mwes)

        base = BPEScheme.learn(data, vocab_size)
        term_freqs, nlines = WordScheme.term_frequencies(data)
        mwes_list = {}

        ngrams_lists, _ = cls.get_ngrams_lists(data, 
                            [x for x in mwes if type(x)==int],
                            ngram_sorter, min_freq, bpe_vocab=base)
        sgrams_lists, _ = cls.get_sgrams_lists(data, 
                            [x for x in mwes if type(x)!=int],
                            skipgram_sorter, min_freq, min_instances,
                            bpe_vocab=base)
        
        mwes_list.update(ngrams_lists)
        mwes_list.update(sgrams_lists)

        if len(toks_list) == 0:
            toks_list = [max_mwes // len(mwes)] * len(mwes)
        assert vocab_size > sum(toks_list)

        return cls.merge_lists(base, mwes_list, vocab_size,
                                mwes, toks_list)


class ExtMWEScheme(BPEScheme):

    PLACE_TOK = '▂', 6 # U+2582 ??
    SKIP_TOK = '<skp>', 7
    TOKS = [PLACE_TOK, SKIP_TOK]
    
    space_char = Reseved.SPACE_TOK[0]
    skip_char = PLACE_TOK[0]
    skip_tok = SKIP_TOK[0]

    def __init__(self, table:List['Type']):
        log.info('Loading Ext MWE Scheme ...')
        super().__init__(table=table)

        # ssi : skip start index
        # ssi = self._get_skips_start_index(table)

        # regular token trie
        self.root = self.make_vocab_prefix_trie(self.table)
        
        ## Either use a seperate trie for the skipgrams
        
        assert self.unk_idx

    def encode(self, line: str, split_ratio: float = 0.) -> List[int]:
        pieces = self.encode_str(line, split_ratio=split_ratio)
        return [self.str_to_idx.get(piece, self.unk_idx) for piece in pieces]

    def encode_str(self, line:str, split_ratio: float - 0.) -> List[str]:
        
        # First pass : Exactly similar to BPE
        pieces = super().encode_str(line, split_ratio)

        # Second pass : Merge skipgrams from pieces
        possible_skips = set()
        for i in range(len(pieces)-1):
            piece = self.skip_char.join([ pieces[i], pieces[i+2] ])
            sub_pieces = super().encode_str(piece, split_ratio)
            if len(sub_pieces) == 1:
                possible_skips.add(i)

        final_pieces = []
        already_added = set()
        for i, piece in enumerate(pieces):
            if i in already_added:
                final_pieces.append(self.skip_tok)
                continue

            if i not in possible_skips:
                final_pieces.append(piece)
            else:
                final_pieces.append(self.skip_char.join( [piece , pieces[i+2]] ))
                already_added.add(i+2)

        return final_pieces

    def decode_str(self, seq : List[str])-> str:
        
        decoded_seq = []
        to_add = set()
        tok_to_add = dict()

        idx = 0
        pos = 0
        while idx < len(seq):
            if pos in to_add:
                decoded_seq.append(tok_to_add[pos])
                to_add.remove(pos)
                pos += 1
                continue

            if self.skip_char not in seq[idx]:
                decoded_seq.append(seq[idx])
            else:
                toks = seq[idx].split(self.skip_char)
                decoded_seq.append(toks[0])
                to_add.add(pos+2)
                tok_to_add[pos+2]=toks[1]

            idx += 1
            pos += 1

        if len(to_add) != 0:
            decoded_seq.append(self.skip_tok)
            decoded_seq.append(tok_to_add[pos+1])

        return ''.join(decoded_seq).replace(self.space_char, ' ')

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, 
            global_list_file:Union[str, Path]=None,
            max_mwes:int=-1):
        base = BPEScheme.learn(data, vocab_size)

        if global_list_file is None:
            return base

        bis, tris, skips = cls.load_global_list(global_list_file)

        ## When the value for max_mwes is not set up we will replace all the
        ## tokens that we can ( keeping the sorted by the frequency maintained )
        ## When the value is set we will check how many tokens can be replaced 
        ## ( the number may be less than the specified by the number )
        ## Should we add the mwes at the end in the first case as well ??
        ## Process scan through all the lists and create a shortened list of tokens
        ## that will be included in the list. ( when no value is set ) and then if the
        ## value is set then we can merge the mwes by the frequency.

        filtered_bis , filtered_tris , filtered_skips = [], [], []

        count = 0
        idx, bidx, tidx, sidx = 0,0,0,0

        while count < vocab_size:
            max_freq = max(base[idx].freq, bis[bidx].freq , tris[tidx].freq, skips[sidx].freq)
            # Ask ( which token to get preference here ) ??
            if max_freq == skips[sidx].freq:
                sidx += 1
            elif max_freq == tris[tidx].freq:
                tidx += 1
            elif max_freq == bis[bidx].freq:
                bidx += 1
            else:
                idx += 1
            count += 1

        # Do we need this hyper parameter as of now ??
        if max_mwes != -1:
            pass

        mwes = []
        mwes.update(base[:idx])
        mwes.update(bis[:bidx])
        mwes.update(tris[:tidx])
        mwes.update(skips[:sidx])

        assert len(mwes) == vocab_size , "The new prepped MWEs is of length <  vocab_size"

        idx = 0
        for i in len(mwes):
            mwes[i].idx = idx
            idx += 1

        return mwes

    @classmethod
    def load_global_list(cls, filepath:Union[str, Path]):
        global_lists = json.loads(filepath.read_text())
        bis = cls._make_ngram_types(global_lists['bi'])
        tris = cls._make_ngram_types(global_lists['tris'])
        skips = cls._make_skip_types(global_lists['skips'])
        return bis, tris, skips

    @classmethod
    def _make_ngram_types(cls, ngram_list:List[List[str],int]):
        types = []
        idx = 0
        for toks, freq in ngram_list:
            name = cls.space_char.join(toks) + cls.space_char
            types.append(Type(name, level=3, freq=freq, idx=idx))
            idx += 1
        return types

    @classmethod
    def _make_skip_types(cls, skips_list:List[List[str], int]):
        types = []
        idx = 0
        for toks, freq in skips_list:
            words = [ x + cls.space_char for x in toks ]
            name = cls.skip_char.join(words)
            types.append(Type(name, level=6, freq=freq, idx=idx))
            idx += 1
        return types

    def _get_skips_start_index(self, table:List['Type']):
        start, end = 0, len(table)-1
        while start < end:
            if end - start == 1:
                break
            mid = ( start + end ) // 2
            name = table[mid].name
            if self.skip_char in name:
                end = mid
            else:
                start = mid
        return end

#########################
REGISTRY = {
    'char': CharScheme,
    'word': WordScheme,
    'bpe': BPEScheme,
    'subword': BPEScheme,
    'class': ClassScheme,
    'ngram': NgramScheme,
    'skipgram': SkipScheme,
    'mwe': MWEScheme,
    'extmwe' : ExtMWEScheme
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
        levels = [CharScheme, BPEScheme, WordScheme, ClassScheme, NgramScheme, SkipScheme, ExtMWEScheme]
        assert max_level < len(levels)
        log.info(f'Max Level for Vocab : {max_level}')
        Scheme = levels[max_level]
    return Scheme(table=types)


def get_scheme(pieces:str):
    if pieces in REGISTRY.keys():
        return REGISTRY[pieces]
    return ValueError(f'Piece {pieces} not available. \
                Choices : [ char, word, bpe, ngram, skipgram, mwe, extmwe ]')


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
