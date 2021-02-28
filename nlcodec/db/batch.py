#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/29/20
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator, Iterable, Optional, Tuple, Union

import numpy as np

from nlcodec import log
from .core import Db, MultipartDb

Array = np.ndarray


class IdExample:
    """
    An object of this class holds an example in sequence to sequence dataset
    """
    __slots__ = ('x', 'y', 'id', 'x_raw', 'y_raw')

    def __init__(self, id, x, y):
        self.x: Array = x
        self.y: Array = y
        self.id = id
        self.x_raw: Optional[str] = None
        self.y_raw: Optional[str] = None

    def val_exists_at(self, side, pos: int, exist: bool, val: int):
        assert side == 'x' or side == 'y'
        assert pos == 0 or pos == -1
        seq = self.x if side == 'x' else self.y
        if exist:
            if seq[pos] != val:
                if pos == 0:
                    seq = np.append(np.int32(val), seq)
                else:  # pos = -1
                    seq = np.append(seq, np.int32(val))
                # update
                if side == 'x':
                    self.x = seq
                else:
                    self.y = seq
        else:  # should not have val at pos
            assert seq[pos] != val

    def eos_check(self, side, exist):
        raise

    def __getitem__(self, key):
        if key == 'x_len':
            return len(self.x)
        elif key == 'y_len':
            return len(self.y)
        else:
            return getattr(self, key)


@dataclass
class BatchMeta:
    pad_idx: int
    bos_idx: int
    eos_idx: int
    add_bos_x: bool = False
    add_bos_y: bool = False
    add_eos_x: bool = True
    add_eos_y: bool = True


class Batch:
    """
    An object of this class holds a batch of examples
    """
    _x_attrs = ('x_len', 'x_seqs')
    _y_attrs = ('y_len', 'y_seqs')

    def __init__(self, batch: List[IdExample], sort_dec=False, batch_first=True,
                 meta: BatchMeta = None):
        """
        :param batch: List fo Examples
        :param sort_dec: True if the examples be sorted as descending order of their source sequence lengths
        :Param Batch_First: first dimension is batch
        """
        assert isinstance(meta, BatchMeta)
        self.meta = meta
        self.batch_first = batch_first

        self.bos_eos_check(batch, 'x', meta.add_bos_x, meta.add_eos_x)
        if sort_dec:
            batch = sorted(batch, key=lambda _: len(_.x), reverse=True)
        self._len = len(batch)
        self.x_len = np.array([len(e.x) for e in batch])
        self.x_toks = np.sum(self.x_len)
        self.max_x_len = np.max(self.x_len)

        self.x_seqs = np.full(shape=(self._len, self.max_x_len), fill_value=self.meta.pad_idx,
                              dtype=int)
        for i, ex in enumerate(batch):
            self.x_seqs[i, :len(ex.x)] = ex.x
        if not batch_first:  # transpose
            self.x_seqs = np.transpose(self.x_seqs)
        self.x_raw = None
        if batch[0].x_raw:
            self.x_raw = [ex.x_raw for ex in batch]

        first_y = batch[0].y
        self.has_y = first_y is not None
        if self.has_y:
            self.bos_eos_check(batch, 'y', meta.add_bos_y, meta.add_eos_y)
            self.y_len = np.array([len(e.y) for e in batch])
            self.y_toks = np.sum(self.y_len)
            self.max_y_len = np.max(self.y_len)
            self.y_seqs = np.full(shape=(self._len, self.max_y_len), fill_value=meta.pad_idx,
                                  dtype=int)
            for i, ex in enumerate(batch):
                self.y_seqs[i, :len(ex.y)] = ex.y

            if not batch_first:  # transpose
                self.y_seqs = np.transpose(self.y_seqs)
            self.y_raw = None
            if batch[0].y_raw:
                self.y_raw = [ex.y_raw for ex in batch]

    @staticmethod
    def val_exists(obj, field_name, pos: int, exist: bool, val: int):
        assert pos == 0 or pos == -1
        seq = getattr(obj, field_name)
        if exist:
            if seq[pos] != val:
                if pos == 0:
                    seq = np.append(np.int32(val), seq)
                else:  # pos = -1
                    seq = np.append(seq, np.int32(val))
                # update
                setattr(obj, field_name, seq)
        else:  # should not have val at pos
            assert seq[pos] != val

    def bos_eos_check(self, batch: List[IdExample], side: str, bos: bool, eos: bool):
        """
        ensures and inserts (if needed) EOS and BOS tokens
        :param batch:
        :param side: which side? choices: {'x', 'y'}
        :param bos: True if should have BOS, False if should not have BOS
        :param eos: True if should have EOS, False if should not have EOS
        :return: None, all modifications are inplace of batch
        """
        for ex in batch:
            self.val_exists(ex, field_name=side, pos=0, exist=bos, val=self.meta.bos_idx)
            self.val_exists(ex, field_name=side, pos=-1, exist=eos, val=self.meta.eos_idx)

    def __len__(self):
        return self._len

    def to(self, device):
        """Move this batch to given device"""
        for name in self._x_attrs + (self._y_attrs if self.has_y else []):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def make_autoreg_mask(self, tgt):
        "Create a mask to hide padding and future words for autoregressive generation."
        return self.make_autogres_mask_(tgt, self.meta.pad_idx)

    @staticmethod
    def make_autogres_mask_(seqs, pad_val: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (seqs != pad_val).unsqueeze(1)
        tgt_mask = tgt_mask & subsequent_mask(seqs.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def subsequent_mask(size):
    """
    Mask out subsequent positions. upper diagonal elements should be zero
    :param size:
    :return: mask where positions are filled with zero for subsequent positions
    """
    # upper diagonal elements are 1s, lower diagonal and the main diagonal are zeroed
    ones = np.ones((size, size), dtype=np.int8)
    triu = np.triu(ones, k=1)
    # invert it
    mask = triu == 0
    mask = mask.unsqueeze(0)
    return mask


class BatchIterable(Iterable[Batch]):

    # This should have been called as Dataset
    def __init__(self, data_path: Path, batch_size: Union[int, Tuple[int, int]], batch_meta: BatchMeta,
                 sort_desc: bool = False, batch_first: bool = True, sort_by: str = None,
                 keep_in_mem=False):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of tokens on the target size per batch; or (max_toks, max_sents)
        :param sort_desc: should the mini batch be sorted by sequence len (useful for RNN api)
        :param keep_in_mem: keep all parts in memory for multipartdb;
           for single part, of course, the part remains in memory.
        """
        self.batch_meta = batch_meta
        self.sort_desc = sort_desc
        if isinstance(batch_size, int):
            self.max_toks, self.max_sents = batch_size, float('inf')
        else:
            self.max_toks, self.max_sents = batch_size
        self.batch_first = batch_first
        self.sort_by = sort_by
        self.data_path = data_path
        self.keep_in_mem = keep_in_mem
        assert sort_by in (None, 'eq_len_rand_batch', 'random')
        if not isinstance(data_path, Path):
            data_path = Path(data_path)

        assert data_path.exists(), f'Invalid State: {data_path} is NOT found.'
        if data_path.is_file():
            self.data = Db.load(data_path, rec_type=IdExample)
        elif self.data_path.is_dir():
            self.data = MultipartDb.load(data_path, rec_type=IdExample, keep_in_mem=keep_in_mem)
        else:
            raise Exception(f'Invalid State: {data_path} is should be a file or dir.')

        log.info(f'Batch Size = {batch_size}, sort_by={sort_by}')

    def read_all(self):
        batch = []
        max_len = 0
        for ex in self.data:
            if min(len(ex.x), len(ex.y)) == 0:
                log.warning("Skipping a record,  either source or target is empty")
                continue

            this_len = max(len(ex.x), len(ex.y))
            if (len(batch) + 1) * max(max_len, this_len) <= self.max_toks and len(batch) < self.max_sents :
                batch.append(ex)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.max_toks:
                    raise Exception(f'Unable to make a batch of {self.max_toks} toks'
                                    f' with a seq of x_len:{len(ex.x)} y_len:{len(ex.y)}')
                # yield the current batch
                yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                            meta=self.batch_meta)
                batch = [ex]  # new batch
                max_len = this_len
        if batch:
            log.debug(f"\nLast batch, size={len(batch)}")
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                        meta=self.batch_meta)

    def make_eq_len_ran_batches(self):
        batches = self.data.make_eq_len_ran_batches(max_toks=self.max_toks, max_sents=self.max_sents)
        for batch in batches:
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                        meta=self.batch_meta)

    def __iter__(self) -> Iterator[Batch]:
        if self.sort_by == 'eq_len_rand_batch':
            yield from self.make_eq_len_ran_batches()
        else:
            yield from self.read_all()

    @property
    def num_items(self) -> int:
        return len(self.data)

    @property
    def num_batches(self) -> int:
        return int(math.ceil(len(self.data) / self.max_toks))
