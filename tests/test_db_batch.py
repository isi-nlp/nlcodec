import random
import shutil
from pathlib import Path

import numpy as np

from nlcodec.db.batch import BatchMeta, BatchIterable
from nlcodec.db.core import Db, MultipartDb

batch_meta = BatchMeta(pad_idx=0, bos_idx=1, eos_idx=2, add_bos_x=True, add_eos_x=True,
                       add_bos_y=True, add_eos_y=True)


def get_data(n_recs, vocab_size):

    for i in range(n_recs):
        _len = np.random.randint(4, 40)
        x = np.random.randint(0, vocab_size, _len)
        y = np.random.randint(0, vocab_size, _len + random.randint(0, 5))
        if batch_meta.add_bos_x:
            x[0] = batch_meta.bos_idx
        if batch_meta.add_eos_x:
            x[-1] = batch_meta.eos_idx
        if batch_meta.add_bos_y:
            y[0] = batch_meta.bos_idx
        if batch_meta.add_eos_y:
            y[-1] = batch_meta.eos_idx
        yield x, y


def prod(*args):
    p = 1
    for a in args:
        p *= a
    return p


def test_batch():
    path = Path('tmp.test.db.batch')
    if not path.exists():
        recs = list(get_data(n_recs=20_000, vocab_size=32_000))
        db = Db.create(recs=recs, field_names=['x', 'y'])
        db.save(path)
    try:
        batch_size = 2000
        bs = BatchIterable(data_path=path, batch_size=2000, batch_meta=batch_meta,
                           sort_by='eq_len_rand_batch')
        count = 0
        for b in bs:
            count += 1
            assert b.y_toks <= batch_size
    finally:
        path.unlink()
        pass


def test_multipart_db_batch():
    path = Path('tmp.test.multidb.batch')

    if not path.exists():
        n_recs = 100_000
        recs = list(get_data(n_recs=n_recs, vocab_size=32_000))
        MultipartDb.create(path=path, recs=recs, field_names=['x', 'y'], part_size=n_recs//10)
    try:
        batch_size = 2000
        bs = BatchIterable(data_path=path, batch_size=batch_size, batch_meta=batch_meta,
                           sort_by='eq_len_rand_batch')
        count = 0
        for b in bs:
            count += 1
            assert b.y_toks <= batch_size
    finally:
        shutil.rmtree(path)
        pass
