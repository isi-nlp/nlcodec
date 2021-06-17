#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/20


import itertools
import json
import math
import os
import pickle
import random
import shutil
from collections import namedtuple, defaultdict
from pathlib import Path
from typing import List, Iterator, Dict, Any, Tuple
import copy
import random
import numpy as np

from nlcodec import log
from nlcodec.utils import as_path

Array = np.ndarray
Record = Tuple[Array]

DEF_TYPE = np.uint16  # uint16 is [0, 65,535]
DEF_MIN = np.iinfo(DEF_TYPE).min
DEF_MAX = np.iinfo(DEF_TYPE).max
DEF_PART_SIZE = 5_000_000
DEF_MAX_PARTS = 1_000


def best_dtype(mn, mx):
    """
    determines best (integer) data type for a given range of integer
    :param mn: min value
    :param mx: max value
    :return: numpy data type
    """
    assert isinstance(mn, (int, np.integer))
    assert isinstance(mx, (int, np.integer))
    # smaller to bigger
    options = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]
    found_type = None
    for int_type in options:
        info = np.iinfo(int_type)
        if info.min <= mn and mx <= info.max:  # completely fits inside
            found_type = int_type
            break
    assert found_type, f'Could not find an integet type for [{min},{max}]'
    return found_type


def part_path_pads(n_parts: int):
    return math.ceil(math.log10(n_parts))


class SeqField:
    """A field of unequal length sequences"""

    class Builder:
        """
        Writer to assist creation of
        """
        # 100MB; each int takes 28 bytes + 8 byte for ref
        BUFFER_SIZE = 100_000_000 / (28 + 8)

        def __init__(self, name, buf_size=None):

            self.name = name
            self.ids = {}
            self.frozen = np.array([], dtype=np.int8)   # start with smallest type
            self.buffer = []
            self.refs = []
            self.max_len = 0
            self.buf_size = buf_size or self.BUFFER_SIZE

        def append(self, id, arr):
            assert len(arr) == 0 or isinstance(arr[0], (int, np.integer))
            self.ids[id] = len(self.refs)
            self.max_len = max(len(arr), self.max_len)
            self.refs.append([len(self.frozen) + len(self.buffer), len(arr)])    # start, len
            self.buffer.extend(arr)
            if len(self.buffer) >= self.buf_size:
                self.shrink()
            return self

        def shrink(self):
            if self.buffer:
                dtype = best_dtype(mn=min(self.buffer), mx=max(self.buffer))
                data = np.array(self.buffer, dtype=dtype)
                self.frozen = np.concatenate((self.frozen, data))
                self.buffer = []

        def build(self):
            self.shrink()
            assert not self.buffer
            refs_type = best_dtype(mn=0, mx=max(len(self.frozen), self.max_len))
            return SeqField(name=self.name, ids=self.ids,
                            refs=np.array(self.refs, dtype=refs_type), data=self.frozen)

    def __init__(self, name: str, ids: Dict[Any, int], refs: Array, data: Array):
        self.name = name
        self.ids = ids
        self.refs = refs
        self.data = data
        assert len(data.shape) == 1
        assert refs.shape == (len(ids), 2)
        assert refs[-1][0] <= len(data)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        start, length = self.refs[self.ids[item]]
        return self.data[start: start + length]

    def get_len(self, _id):
        return self.refs[self.ids[_id]][1]

    def keys(self):
        return self.ids.keys()

    def values(self):
        for start, _len in self.refs:
            yield self.data[start: start + _len]

    def items(self):
        for id, idx in self.ids.items():
            start, _len = self.refs[idx]
            yield self.data[start: start + _len]

    def lengths(self):
        return ((id, self.refs[idx, 1]) for id, idx in self.ids.items())

    @classmethod
    def create(cls, name, recs) -> 'SeqField':
        builder = cls.Builder(name=name)
        for id, arr in recs:
            builder.append(id, arr)
        return builder.build()

    @classmethod
    def create_many(cls, names: List[str], recs: Iterator[List[List[int]]]) -> List['SeqField']:
        builders = [cls.Builder(name) for name in names]
        for id, rec in recs:
            assert len(rec) == len(builders)
            for b, col in zip(builders, rec):
                b.append(id, col)
        return [b.build() for b in builders]


class Db:

    def __init__(self, fields: List[SeqField], rec_type=None, shuffle=False):
        assert all(isinstance(f, SeqField) for f in fields)
        self.field_names = [f.name for f in fields]
        self._rec_type = rec_type
        self.fields = {f.name: f for f in fields}
        self._len = len(fields[0])
        self.ids = set(fields[0].ids.keys())
        self.shuffle = shuffle
        for fn, fd in self.fields.items():
            assert self._len == len(fd)  # all have same num of recs
            assert self.ids == set(fd.ids.keys())  # and same IDs

    # noinspection PyPep8Naming
    def RecType(self):
        if not self._rec_type:
            self._rec_type = namedtuple('RecType', ['id'] + self.field_names)
        return self._rec_type

    def __len__(self):
        return self._len

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_rec_type'] = None  # bcoz it is not pickleable
        return state

    def save(self, path):
        log.debug(f"Saving to {path}")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path, rec_type=None, shuffle=False) -> 'Db':
        log.debug(f"Loading from {path}: shuffle={shuffle}")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        obj.shuffle = shuffle
        if rec_type:
            obj._rec_type = rec_type
        return obj

    @classmethod
    def create(cls, recs, field_names, has_id=False, path=None):
        """
        :param recs: Iterator[List[List[int]]] or Iterator[(id, List[List[int]])]
        :param field_names: field names in records
        :param has_id: set True if recs already have id ie. tuple(id, fields),
           when set to False (default) ids are auto generated
        :param path: path to save on disk (optional)
        :return:
        """
        if not has_id:
            recs = enumerate(recs)
        fields = SeqField.create_many(field_names, recs)
        db = cls(fields=fields)
        if path:
            db.save(path)
        return db

    def __getitem__(self, _id):
        cols = tuple(self.fields[fn][_id] for fn in self.field_names)
        return self.RecType()(_id, *cols)

    def __iter__(self):
        ids = self.ids
        if self.shuffle:
            ids = copy.copy(ids)
            random.shuffle(ids)
        for _id in ids:
            yield self[_id]

    def _make_eq_len_batch_ids(self, max_toks, max_sents, min_len=1):
        fields = list(self.fields.values())
        rows = []
        skip_counter = defaultdict(int)
        for _id in self.ids:
            lens = tuple(field.get_len(_id) for field in fields)
            if min(lens) < min_len:
                skip_counter[f'len < {min_len}'] += 1
            else:
                rows.append((_id, max(lens)))
        if len(skip_counter) > 0:
            log.warning(f"Skipped :: {skip_counter}")
        rows = np.array(rows)  # id, len
        np.random.shuffle(rows)  # in-place, along the first axis; for extra rand within len group
        rows = rows[rows[:, 1].argsort()]  # sort by second col wiz len
        batches = []
        batch = []
        max_len = 0
        for _id, _len in rows:
            if _len < 1:
                log.warning(f"Skipping record {_id}, either source or target is empty")
                continue

            if (len(batch) + 1) * max(max_len, _len) > max_toks or len(batch) > max_sents:
                if _len > max_toks:
                    raise Exception(f'Unable to make a batch of {max_toks} toks'
                                    f' with a seq of len:{_len}')
                batches.append(np.array(batch))
                batch = []  # new batch
                max_len = 0

            batch.append(_id)  # this one can go in
            max_len = max(max_len, _len)
        if batch:
            batches.append(np.array(batch))
        return batches

    def make_eq_len_ran_batches(self, max_toks, max_sents=float('inf')):

        batches = self._make_eq_len_batch_ids(max_toks=max_toks, max_sents=max_sents)
        if not batches:
            raise Exception(f'Found no data. Please check config data paths')
        log.info(f"length sorted random batches = {len(batches)}. Shuffling🔀...")
        # every pass introduce some randomness
        random.shuffle(batches)

        for batch_ids in batches:
            batch = [self[_id] for _id in batch_ids]
            yield batch


class MultipartDb:

    @classmethod
    def slices(cls, stream, size):
        _stream = iter(stream)
        try:
            while True:
                segment = itertools.islice(_stream, size)
                n = next(segment)  # so that it raises StopIteration when empty, to break the loop
                yield itertools.chain([n], segment)
        except StopIteration:
            pass

    @classmethod
    def create(cls, path, recs, field_names, has_id=False, overwrite=False,
               part_size=DEF_PART_SIZE, max_parts=DEF_MAX_PARTS):
        if not has_id:
            recs = enumerate(recs)
        builder = cls.Writer(path=path, field_names=field_names, overwrite=overwrite,
                             max_parts=max_parts)

        part_num = -1
        for sliced in cls.slices(recs, part_size):
            part_num += 1
            builder(part_num, recs=sliced)
        builder.close()
        return cls.load(path=path)

    @classmethod
    def load(cls, path, rec_type=None, keep_in_mem=False, shuffle=False) -> 'MultipartDb':
        path = as_path(path)
        assert path.is_dir()
        flag_file = path / '_SUCCESS'
        assert flag_file.exists(), 'Looks like the save was incomplete or corrupted'
        meta = json.loads(flag_file.read_text())
        part_files = []
        rec_counts = []
        for part_name, stats in meta['parts'].items():
            part_file = path / part_name
            part_file.exists()
            part_files.append(part_file)
            rec_counts.append(stats['count'])
        field_names = meta['field_names']
        rec_type = rec_type or namedtuple('RecType',  ['id'] + field_names)
        assert len(part_files) > 0
        return cls(parts=part_files, rec_counts=rec_counts, rec_type=rec_type,
                   keep_in_mem=keep_in_mem, shuffle=shuffle)

    def __init__(self, parts: List[Path], rec_counts: List[int], rec_type, keep_in_mem=False,
                 shuffle=False):
        self.part_paths = parts
        self.rec_counts = rec_counts
        self._len = sum(rec_counts)
        self.rec_type = rec_type
        self.keep_in_mem = keep_in_mem or len(parts) == 1  # if its only one part, just keep it
        self.shuffle =  shuffle
        self.mem = [None] * len(parts)
        if self.keep_in_mem:
            self.mem = [Db.load(p, rec_type=rec_type) for p in parts]

    def __len__(self):
        return self._len

    def __iter__(self):
        idx_paths = list(enumerate(self.part_paths))
        if self.shuffle:
            random.shuffle(idx_paths)
        for idx, path in idx_paths:
            if self.keep_in_mem:
                part = self.mem[idx]
            else:
                part = Db.load(path, shuffle=self.shuffle)
            yield from part

    def make_eq_len_ran_batches(self, max_toks, max_sents=float('inf'), join_ratio=0.0) -> Iterator[List]:
        # shuffle the parts
        buff = list(zip(self.part_paths, self.mem))
        random.shuffle(buff)

        for path, part in buff:
            if part is None:
                part = Db.load(path, rec_type=self.rec_type)
            yield from part.make_eq_len_ran_batches(max_toks=max_toks, max_sents=max_sents)

    class Writer:

        def __init__(self, path, field_names: List[str], overwrite=False,
                     max_parts=DEF_MAX_PARTS):
            self.field_names = field_names
            path = as_path(path)
            if path.exists() and len(os.listdir(path)) > 0:
                if overwrite:
                    log.warning(f"Removing existing data at {path}")
                    shutil.rmtree(path)
                else:
                    raise Exception(f'{path} already exists. not overwriting it')
            path.mkdir(parents=True, exist_ok=True)
            self.path = path
            self.part_path_pad = part_path_pads(max_parts)

        def __call__(self, part_num: int, recs):
            # assume recs have ids created externally
            part = Db.create(recs, field_names=self.field_names, has_id=True)
            part_path = self.path / f'part-{part_num:0{self.part_path_pad}d}'
            part.save(part_path)
            meta_path = part_path.with_suffix('.meta')
            meta = dict(count=len(part), size=part_path.stat().st_size)
            meta = json.dumps(meta, ensure_ascii=False, indent=2)
            meta_path.write_text(meta)
            return part_path, len(part)

        def close(self):
            parts = {}
            for meta_path in self.path.glob("part-*.meta"):
                part_name = meta_path.name.rstrip('.meta')
                parts[part_name] = json.loads(meta_path.read_text())
            meta = dict(parts=parts, field_names=self.field_names)
            flag_file = self.path / '_SUCCESS'
            flag_file.write_text(json.dumps(meta, indent=2))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_tb is None:
                self.close()
                return True
            else:
                return False
