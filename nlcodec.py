#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-16

import abc

import argparse
import sys
from pathlib import Path
import collections as coll
from typing import List, TextIO, Dict, Tuple, Union, Iterator, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass
import json
from datetime import datetime
import logging as log

# my own custom data structures
from dstruct import TrNode

log.basicConfig(level=log.INFO)

__version__ = 0.1
Codes = Dict[int, Tuple[int, ...]]
Seq = List[int]
Bigram = Tuple[int, int]


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
    def new(cls, types: List[str]):
        res = [tok for tok, idx in cls.ALL]
        if types:
            assert len(set(types)) == len(types)
            assert len(set(types) & set(res)) == 0
            res += types
        return res

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
    user = 0  # as of now,  user and char are indistinguishable; 0 means dont look inside
    subword = 1
    word = 2
    phrase = 3


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
    def write_out(cls, table: List['Type'], out: Union[Path, TextIO]):

        wrtr = out.open('w', encoding='utf8', errors='ignore') if isinstance(out, Path) else out
        levels = dict(coll.Counter(v.level for v in table))
        max_level = max(levels.keys())
        meta = dict(total=len(table), version=__version__, levels=levels, max_level=max_level,
                    created=str(datetime.now()))
        meta = json.dumps(meta)
        wrtr.write(f"#{meta}\n")
        for i, item in enumerate(table):
            assert i == item.idx, f'{item} expected index {i}'
            wrtr.write(item.format() + '\n')
        if wrtr is not out:
            wrtr.close()
        log.info(f"Wrote {len(table)} to {wrtr.name}")

    @classmethod
    def read_vocab(cls, inp: Union[Path, TextIO]) -> Tuple[List['Type'], Optional[Dict]]:
        rdr = inp.open('r', encoding='utf8', errors='ignore') if isinstance(inp, Path) else inp
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


class EncoderScheme:

    def __init__(self, table: List[Type], validate=True):
        if validate:
            Reseved.validate(table)
            self.unk_idx = Reseved.UNK_IDX
        else:
            # at least UNK should be available
            assert table[Reseved.UNK_IDX].name == Reseved.UNK_TOK[0]
            # TODO: reverse lookup UNK IDX based on UNK_TOK name
            self.unk_idx = Reseved.UNK_IDX

        self.vocab_size = len(table)
        self.table = table
        self.idx_to_str = [t.name for t in table]
        self.str_to_idx = {tok: idx for idx, tok in enumerate(self.idx_to_str)}
        assert len(self.idx_to_str) == len(self.str_to_idx)

    def __len__(self):
        return self.vocab_size

    @property
    @classmethod
    def name(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def encode_str(self, line: str) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def decode_str(self, seq: List[str]) -> str:
        raise NotImplementedError()

    def encode(self, line: str) -> List[int]:
        pieces = self.encode_str(line)
        return [self.str_to_idx.get(piece, self.unk_idx) for piece in pieces]

    def decode(self, seq: List[int]) -> str:
        pieces = [self.idx_to_str[idx] for idx in seq]
        return self.decode_str(pieces)

    @classmethod
    @abc.abstractmethod
    def learn(cls, data: Iterator[str], **kwargs) -> List[Type]:
        raise NotImplementedError()


class WordScheme(EncoderScheme):

    @classmethod
    def encode_str(cls, line: str) -> List[str]:
        return line.split()

    @classmethod
    def decode_str(cls, seq: List[str]) -> str:
        return " ".join(seq)

    @property
    @classmethod
    def name(cls):
        return "word"

    @classmethod
    def learn(cls, data: Iterator[str], vocab_size: int=0, **kwargs) -> List[Type]:
        assert not kwargs
        log.info(f"Building {cls} vocab.. This might take some time")
        stats = coll.Counter()
        for line in tqdm(data):
            stats.update(cls.encode_str(line.strip()))
        log.info(f"Found {len(stats):,} types and {sum(stats.values()):,} tokens")

        vocab = Reseved.with_reserved_types()
        for r_type in vocab:
            if r_type.name in stats:
                log.warning(f"Found reserved type {r_type.name} with freq {stats[r_type.name]} ")
                del stats[r_type.name]
        stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        if vocab_size > 0 and len(vocab) + len(stats) > vocab_size:
            log.info(f"truncating vocab at size={vocab_size}")
            stats = stats[:vocab_size - len(vocab)]
        vocab += [Type(name=name, idx=idx, freq=freq, level=Level.char)
                  for idx, (name, freq) in enumerate(stats, start=len(vocab))]
        log.info(f"Total {cls} vocab size {len(vocab):,}")
        return vocab


class CharScheme(WordScheme):
    space_char = '▁'  # U+2581 same as google/sentencepiece

    @property
    def name(self):
        return "char"

    @classmethod
    def encode_str(cls, line: str) -> List[str]:
        return list(cls.space_char.join(line.split()))

    @classmethod
    def decode_str(cls, seq: List[str]) -> str:
        return ''.join(seq).replace(cls.space_char, ' ')

    # learn() is same as parent class: WordScheme

class BPEScheme(CharScheme):

    def __init__(self, table: List[Type]):
        super().__init__(table=table)
        self.root = self.make_vocab_prefix_trie(self.table)
        log.info(f"Vocab size={len(self.table)}; trie root has nodes={self.root.size}"
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

    @property
    def name(self):
        return "bpe"

    def encode_str(self, line: str) -> List[str]:
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

        return [self.table[idx].name for idx in res]

    def decode_str(self, seq: List[str]) -> str:
        return ''.join(seq).replace(self.space_char, ' ').strip()

    @classmethod
    def learn(cls, data: Iterator[str], vocab_size: int=0, **kwargs) -> List[Type]:
        assert vocab_size > 0
        assert not kwargs
        from bpe import BPELearn
        vocab = BPELearn.learn_subwords_from_corpus(corpus=data, vocab_size=vocab_size)
        return vocab


#########################
REGISTRY = {
    'char': CharScheme,
    'word': WordScheme,
    'bpe': BPEScheme,
    'subword': BPEScheme
}


def learn_vocab(inp, level, model, vocab_size):
    log.info(f"Learn Vocab for level={level} and store at {model}")
    log.info(f"data ={inp}")
    Scheme = REGISTRY[level]
    table = Scheme.learn(inp, vocab_size=vocab_size)
    Type.write_out(table=table, out=model)


def load_scheme(path: Path) -> EncoderScheme:
    types, meta = Type.read_vocab(path)
    assert meta
    max_level = meta['max_level']
    levels = [CharScheme, BPEScheme, WordScheme]
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


def write_lines(lines: Iterator[str], out: TextIO, line_break='\n'):
    for line in lines:
        out.write(line)
        out.write(line_break)


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("task", choices=['learn', 'encode', 'decode'], help='task')

    p.add_argument("--level", choices=['char', 'word', 'bpe'],
                   help='Encoding Level; Valid only for "learn" task')

    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path. Not valid for "learn" task')
    p.add_argument('-m', '--model', type=Path, help='Path to model', required=True)
    p.add_argument('-vs', '--vocab_size', type=int, default=-1,
                   help='Vocabulary size. Valid only for task=learn.')
    p.add_argument('-idx', '--indices', action='store_true', default=None,
                   help='Indices instead of strings. Valid for task=encode and task=decode')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose mode. DEBUG log level.')
    args = vars(p.parse_args())
    if args.pop('verbose'):
        log.getLogger().setLevel(level=log.DEBUG)
    return args


def main():
    args = parse_args()
    task = args.pop('task')
    if task == 'learn':
        args.pop('out')      # No output
        args.pop('indices')  # No output
        learn_vocab(**args)
    elif task in ('encode', 'decode'):
        scheme = load_scheme(args.pop('model'))
        inp, out, indices = args['inp'], args['out'], args.get('indices', False)
        if task == 'encode':
            recs = encode(inp, scheme, indices=indices)
            if indices:
                recs = ([str(idx) for idx in seq] for seq in recs)
            recs = (' '.join(seq) for seq in recs)
        else:
            recs = decode(inp, scheme, indices=indices)
        write_lines(recs, out)
    else:
        raise NotImplementedError(task + ' not implemented')


if __name__ == '__main__':
    main()
