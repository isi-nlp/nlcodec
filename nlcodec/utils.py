#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-11-12
import resource
import sys
from pathlib import Path
from typing import List, Any, Iterable, Dict, Tuple, Union
import collections as coll
from nlcodec import log
from tqdm import tqdm
import gzip
import time
from contextlib import contextmanager
from datetime import timedelta


def make_n_grams(sent: List[Any], n):
    assert n > 0
    return [tuple(sent[i: i + n]) for i in range(len(sent) - n + 1)]


def make_n_grams_all(sents: Iterable[List[Any]], n):
    grams = coll.Counter()
    n_sent = 0
    for sent in tqdm(sents, mininterval=1, dynamic_ncols=True):
        grams.update(make_n_grams(sent, n))
        n_sent += 1
    log.info(f"Made {n}-grams: types={len(grams)}; tokens={sum(grams.values())}")
    return grams


def filter_types_coverage(types: Dict[str, int], coverage=1.0) -> Tuple[Dict[str, int], int]:
    assert  0 < coverage <= 1
    tot = sum(types.values())
    includes = {}
    cum = 0
    types  = sorted(types.items(), key=lambda x: x[1], reverse=True)
    for t, f in types:
        cum += f / tot
        includes[t] = f
        if cum >= coverage:
            break
    log.info(f'Coverage={cum:g}; requested={coverage:g}')
    excludes = {ch: ct for ch, ct in types if ch not in includes}
    unk_count = sum(excludes.values())
    log.warning(f'UNKed total toks:{unk_count} types={len(excludes)} from types:{excludes}')
    return includes, unk_count


def as_path(path: Union[str, Path]) -> Path:
    """
    returns an instance of Path, optionally converting string to Path when needed
    :param path: instance of str or Path
    :return: instance of Path
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


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


class IO:
    """File opener and automatic closer"""

    def __init__(self, path, mode='r', encoding=None, errors=None):
        self.path = path if type(path) is Path else Path(path)
        self.mode = mode
        self.fd = None
        self.encoding = encoding if encoding else 'utf-8' if 't' in mode else None
        self.errors = errors if errors else 'replace'

    def __enter__(self):

        if self.path.name.endswith(".gz"):  # gzip mode
            self.fd = gzip.open(self.path, self.mode, encoding=self.encoding, errors=self.errors)
        else:
            if 'b' in self.mode:  # binary mode doesnt take encoding or errors
                self.fd = self.path.open(self.mode)
            else:
                self.fd = self.path.open(self.mode, encoding=self.encoding, errors=self.errors,
                                         newline='\n')
        return self.fd

    def __exit__(self, _type, value, traceback):
        self.fd.close()

    @classmethod
    def reader(cls, path, text=True):
        return cls(path, 'rt' if text else 'rb')

    @classmethod
    def writer(cls, path, text=True, append=False):
        return cls(path, ('a' if append else 'w') + ('t' if text else 'b'))

    @classmethod
    def read_as_stream(cls, paths: List, text=True):
        """
        reads all files as single stream of lines
        :param paths:
        :param text:
        :return:
        """
        for path in paths:
            with cls.reader(path, text=text) as stream:
                yield from stream


@contextmanager
def log_resources(name=""):
    """
    logs time and memory utilized by a code block
    :param name: some name to identify code block
    :return:
    """
    st = time.time()
    st_mem = max_RSS()[1]
    try:
        yield name
    finally:
        delta = timedelta(seconds=time.time() - st)
        et_mem = max_RSS()[1]
        log.info(f"{name} Time: {delta}; Mem: {st_mem} --> {et_mem}")


def add_bool_arg(parser, name, default=False, help=None, nhelp=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f'--{name}', dest=name, action='store_true', help=help)
    group.add_argument(f'--no-{name}', dest=name, action='store_false', help=nhelp or f'See --{name}')
    parser.set_defaults(**{name: default})
