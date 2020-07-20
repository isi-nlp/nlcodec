#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/20


import itertools
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Union, List, Iterator, Dict, Any

import numpy as np
import math

from nlcodec import log

Array = np.ndarray

DEF_TYPE = np.uint16  # uint16 is [0, 65,535]
DEF_MIN = np.iinfo(DEF_TYPE).min
DEF_MAX = np.iinfo(DEF_TYPE).max
DEF_PART_SIZE = 1_000_000
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


def as_path(path: Union[str, Path]) -> Path:
    """
    returns an instance of Path, optionally converting string to Path when needed
    :param path: instance of str or Path
    :return: instance of Path
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


class SeqField:
    """A field of unequal length sequences"""

    class Builder:
        """
        Writer to assist creation of
        """

        def __init__(self, name):
            self.name = name
            self.ids = {}
            self.data = []
            self.refs = []
            self.max_len = 0

        def append(self, id, arr):
            assert len(arr) == 0 or isinstance(arr[0], (int, np.integer))
            self.ids[id] = len(self.refs)
            self.max_len = max(len(arr), self.max_len)
            self.refs.append([len(self.data), len(arr)])  # start, len
            self.data.extend(arr)
            return self

        def build(self):
            dtype = best_dtype(mn=min(self.data), mx=max(self.data))
            refs_type = best_dtype(mn=0, mx=max(len(self.data), self.max_len))
            return SeqField(name=self.name, ids=self.ids,
                            refs=np.array(self.refs, dtype=refs_type),
                            data=np.array(self.data, dtype=dtype))

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

    def __init__(self, fields: List[SeqField]):
        assert all(isinstance(f, SeqField) for f in fields)
        self.fields = fields
        self.field_names = [f.name for f in fields]
        self._len = len(self.fields[0])
        self.ids = set(self.fields[0].ids.keys())
        for i in range(1, len(fields)):
            assert self._len == len(fields[i])  # all have same num of recs
            assert self.ids == set(self.fields[i].ids.keys())  # and same IDs

    def __len__(self):
        return self._len

    def save(self, path):
        log.info(f"Saving to {path}")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> 'Db':
        log.info(f"Loading from {path}")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
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

    def __iter__(self):
        for _id in self.ids:
            yield _id, tuple(f[_id] for f in self.fields)


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
        builder.finish()
        return cls.load(path=path)

    @classmethod
    def load(cls, path) -> 'MultipartDb':
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
        return cls(parts=part_files, rec_counts=rec_counts)

    def __init__(self, parts: List[Path], rec_counts: List[int]):
        self.part_paths = parts
        self.rec_counts = rec_counts
        self._len = sum(rec_counts)

    def __len__(self):
        return self._len

    def __iter__(self):
        for path in self.part_paths:
            part = Db.load(path)
            yield from part

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
            meta = dict(parts=parts)
            flag_file = self.path / '_SUCCESS'
            flag_file.write_text(json.dumps(meta, indent=2))


def main():
    pass


if __name__ == '__main__':
    main()
