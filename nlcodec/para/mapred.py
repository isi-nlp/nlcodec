#!/usr/bin/env python
#
# map-reduce using multiprocessing
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/2/20
import collections as coll
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Iterator, Any, Callable, TypeVar, Generic
import itertools
import logging as log
from functools import partial

log.basicConfig(level=log.DEBUG)

@dataclass
class SingletonException(Exception):
    item: Any


def pair_subsequent(iterator: Iterator, default=None, throw_singleton=False):
    """
    groups subsequent items in to pairs
    :param iterator: stream from which pairs are to be made
    :param default: default value to be used if there is an odd number of items in iterator
    :return:
    """
    last = None
    idx = 0
    for idx, item in enumerate(iterator, start=1):
        if idx % 2 == 0:  # we have a pair
            yield last, item
        last = item
    if throw_singleton:
        # at least one item should be there
        assert idx > 0, 'At least one item was expected but the input was empty'
        if idx == 1:
            raise SingletonException(last)
    if idx % 2 == 1:  # the last item un paired
        yield last, default


T = TypeVar('T')


@dataclass
class IterativeReducer(Generic[T]):
    func: Callable[[T, T], T]  # the reduce function (T, T) -> T
    default: T  # default value is a pairing item for the odd numbered item

    def dispatcher(self, args):
        assert len(args) == 2
        return self.func(args[0], args[1])

    def __call__(self, iterator: Iterator[T], pool=map) -> T:
        try:
            while True:  # break by exception when  all are reduced to singleton
                pairs = pair_subsequent(iterator=iterator, default=self.default,
                                        throw_singleton=True)
                iterator = pool(self.dispatcher, pairs)
                # iterator isn't evaluated since it is lazy. So, the code goes into infinite loop
                # the trick is to evaluate one item at least! aaha moment
                # This is black magic !
                iterator = itertools.chain([next(iterator)], iterator)
        except SingletonException as result:
            # reduced all the way to a singleton element
            return result.item


def ireduce(reduce_func: Callable[[T, T], T], default: T, data: Iterator[T], pool=map):
    ireducer = IterativeReducer(func=reduce_func, default=default)
    return ireducer(data, pool=pool)


@dataclass
class MapReduce:

    mapper: Callable
    reducer: Callable
    default: Any  # default for reducer, to pair the odd item

    def __call__(self, input, map_pool=None, reduce_pool=None):
        mapped = map_pool(self.mapper, input)
        reduced = ireduce(reduce_func=self.reducer, default=self.default, data=mapped,
                          pool=reduce_pool)
        return reduced

    def multiprocess(self, input, map_procs=2, red_procs=2, chunksize=1000):
        """
        runs map reduce on multi processes
        :param input: iterator of input that goes to mapper function
        :param map_procs: number of mapper processes
        :param red_procs: number of reduce processes
        :param chunksize: chunksize for localising data to processes.
         a smaller value consumes less memory but slow, higher value consumes more memory but fast.
        :return: final result from reducer
        """
        # TODO: support process local reduction
        assert map_procs >= 1
        assert red_procs >= 1
        log.info(f"map_processes: {map_procs}, reduce_processes:{red_procs}")

        with mp.get_context("spawn").Pool(processes=map_procs) as map_pool, \
                mp.get_context("spawn").Pool(processes=red_procs) as red_pool:
            map_pooler = partial(map_pool.imap, chunksize=chunksize)
            red_pooler = partial(red_pool.imap, chunksize=chunksize)
            return self(input, map_pool=map_pooler, reduce_pool=red_pooler)


def word_count_line(text):
    """Mapper task : maps line of text into term freqs at the sentence level"""
    return coll.Counter(text.split())


def merge_freqs(tfs1: Dict[str, int], tfs2: Dict[str, int]) -> Dict[str, int]:
    """
    Merges term-frequencies from tfs1 and tfs2 dictionaries.
    This is an inplace operation i.e. It updates the largest dictionary in the argument inplace
    and returns that as result.
    """
    # keep the largest, dump the smallest;
    keep, dump = tfs1, tfs2
    if len(keep) < len(dump):
        keep, dump = dump, keep
    for k, v in dump.items():
        keep[k] = keep.get(k, 0) + v
    return keep


def WordCounter() -> MapReduce:
    return MapReduce(mapper=word_count_line, reducer=merge_freqs, default={})

def main(lines):
    tfs =  WordCounter().multiprocess(lines)
    tfs = sorted(tfs.items(), key=lambda x:x[1], reverse=True)
    for t,f in tfs:
        print(f'{t}\t{f:,}')

if __name__ == '__main__':
    lines = ["hello this is a line 1", "and this is a line 2", "and blah blah"]
    import sys
    #lines = sys.stdin
    main(lines=lines)
