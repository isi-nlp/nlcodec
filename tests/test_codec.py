#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/26/21

from pathlib import Path

import nlcodec

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


def test_class_scheme():
    labels = "A B C A B A A A A A B C D D C C B A A A A A A D".split()
    args = dict(inp=labels, level='class', vocab_size=-1)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / 'model.tsv'
        args['model'] = model_file
        table = nlc.learn_vocab(**args)
        assert len(table) == 4
        table2, meta = nlc.Type.read_vocab(model_file)
        assert len(table2) == len(table)
        table_str = '\n'.join(x.format() for x in table)
        table2_str = '\n'.join(x.format() for x in table2)
        assert table_str == table2_str


def test_byte_scheme():
    args = dict(inp=IO.read_as_stream(paths=[en_txt, fr_txt]), level='byte')
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / 'model.tsv'
        args['model'] = model_file
        table = nlc.learn_vocab(vocab_size=-1, **args)
        table2, meta = nlc.Type.read_vocab(model_file)
        assert len(table2) == len(table)
        table_str = '\n'.join(x.format() for x in table)
        table2_str = '\n'.join(x.format() for x in table2)
        assert table_str == table2_str
        codec = nlc.load_scheme(model_file)
        for s in ['hello, world!?&%^&$#@1235214"\'',
                  "ಕನ್ನಡ ವಿಶ್ವಕೋಶವು ಮೀಡಿಯಾವಿಕಿಯನ್ನು ಬಳಸಿ ಕಟ್ಟಿರುವ ಸ್ವತಂತ್ರ ವಿಶ್ವಕೋಶ.",
                  "维基百科，自由的百科全书"]:
            e = codec.encode_str(s)
            d = codec.decode_str(e)
            assert s == d
            e = codec.encode(s)
            d = codec.decode(e)
            assert s == d


def test_byte_scheme_reserved():
    codec = nlcodec.ByteScheme()
    s = codec.encode_str("hello world")
    s.insert(0, '<s>')
    print(codec.decode_str(s))

