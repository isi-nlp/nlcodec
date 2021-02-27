#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/26/21

from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data'
en_txt = data_dir / 'train.en.tok'
fr_txt = data_dir / 'train.fr.tok'

assert en_txt.exists()
assert fr_txt.exists()

import nlcodec as nlc
from nlcodec.utils import IO
import tempfile


def test_bpe():
    vocab_size = 6000

    args = dict(inp=IO.read_as_stream(paths=[en_txt, fr_txt]),
                level='bpe',
                vocab_size=vocab_size,
                min_freq=1,
                term_freqs=False,
                char_coverage=0.99999,
                min_co_ev=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / 'model.tsv'
        args['model'] = model_file
        table = nlc.learn_vocab(**args)
        assert len(table) == vocab_size
        table2, meta = nlc.Type.read_vocab(model_file)
        assert len(table2) == len(table)
        table_str = '\n'.join(x.format() for x in table)
        table2_str = '\n'.join(x.format() for x in table2)
        assert  table_str == table2_str


def test_shrink():
    vocab_size = 6000
    args = dict(inp=IO.read_as_stream(paths=[en_txt, fr_txt]),
                level='bpe',
                vocab_size=vocab_size,
                min_freq=1,
                term_freqs=False,
                char_coverage=0.99999,
                min_co_ev=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_file = tmpdir / 'model.tsv'
        en_model_file = tmpdir / 'model.en.tsv'
        args['model'] = model_file
        table = nlc.learn_vocab(**args)
        assert len(table) == vocab_size
        scheme = nlc.load_scheme(model_file)
        mapping = scheme.shrink_vocab(files=[en_txt], min_freq=1, save_at=en_model_file)
        assert len(mapping) > 0
        model2 = nlc.load_scheme(en_model_file)
        assert len(model2.table) == len(mapping)
