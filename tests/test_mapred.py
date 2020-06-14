#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/2/20
from nlcodec.para.mapred import merge_freqs, SingletonException, pair_subsequent, \
    MapReduce


def test_pair_subsequent():
    odd = [1, 2, 3, 4, 5]
    assert [(1, 2), (3, 4), (5, -1)] == list(pair_subsequent(iter(odd), default=-1))

    even = [1, 2, 3, 4, 5, 6]
    assert [(1, 2), (3, 4), (5, 6)] == list(pair_subsequent(iter(even), default=-1))

    empty = []
    assert [] == list(pair_subsequent(iter(empty), default=-1))
    # throw errror when empty
    try:
        list(pair_subsequent(iter(empty), default=-1, throw_singleton=True))
        assert False  # this shouldn't happen
    except AssertionError:
        assert True  # good

    # Collect singleton result from exception
    val = 100
    try:
        list(pair_subsequent(iter([val]), default=-1, throw_singleton=True))
        assert False  # this shouldn't happen
    except SingletonException as e:
        assert e.item == val  # good


def test_merge():
    d1 = dict(a=1, b=2)
    d2 = dict(b=1, c=10, d=20)
    d3 = merge_freqs(d1, d2)
    assert d3 == dict(a=1, b=3, c=10, d=20)
    assert d3 is d2  # picked the largest dict


def _mapper_func(x, overhead=10):
    # some overhead work
    delta = sum(x for x in range(overhead))
    return x + delta - delta


def _reducer_func(a, b, overhead=10):
    # addition plus some overhead work
    delta = sum(x for x in range(overhead))
    return a + b +  delta - delta


def test_map_reduce():
    import time, datetime
    chunksize=10_000
    for i in [1, 10, 1000]:
        n = i * chunksize
        items = range(1, n + 1)

        truth = int(n * (n + 1) * 0.5)
        mapred = MapReduce(mapper=_mapper_func, reducer=_reducer_func, default=0)

        st = time.time()
        base_total = mapred(items, map_pool=map, reduce_pool=map)
        et = time.time()
        print(f"n = {n:,}; single thread; time taken: {datetime.timedelta(seconds=et - st)}")
        assert base_total == truth

        items = range(1, n + 1)
        n_proc = 4
        st = time.time()
        mp_total = mapred.multiprocess(items, chunksize=chunksize)
        et = time.time()
        print(f"n = {n:,}; multiprocess={n_proc}; time taken: {datetime.timedelta(seconds=et - st)}")

        assert mp_total == truth


if __name__ == '__main__':
    test_map_reduce()
