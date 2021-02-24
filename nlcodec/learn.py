#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/31/20

import logging as log
from typing import Dict, Any
from pathlib import Path
from nlcodec import DEF_WORD_MIN_FREQ, DEF_CHAR_MIN_FREQ, DEF_CHAR_COVERAGE, DEF_MIN_CO_EV
from nlcodec import __version__, learn_vocab
from nlcodec.term_freq import word_counts, write_stats
from nlcodec import utils
from nlcodec import spark
from nlcodec import __version__, __epilog__

log.basicConfig(level=log.INFO)

__description__ = 'learn BPE on pyspark'


def parse_args() -> Dict[str, Any]:
    import argparse
    # noinspection PyTypeChecker
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                prog='nlcodec-learn',
                                description=__description__,
                                epilog=__epilog__)
    p.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')

    p.add_argument('-i', '--inp', type=Path, nargs='+',
                   help='Paths to all the input files from which to extract terms.')
    p.add_argument('-m', '--model', type=Path, help='Path to model aka vocabulary file',
                   required=True)

    p.add_argument('-vs', '--vocab-size', type=int, default=-1,
                   help='Vocabulary size. This is required for'
                        ' "bpe", but optional for "word" and "char" models, specifying it'
                        ' will trim the vocabulary at given top most frequent types.')
    p.add_argument('-l', '--level', choices=['char', 'word', 'bpe'], default='bpe',
                   help='Encoding Level')
    p.add_argument('-mf', '--min-freq', default=None, type=int,
                   help='Minimum frequency of types for considering inclusion in vocabulary. '
                        'Types fewer than this frequency will be ignored. '
                        f'For --level=word or --level=bpe, freq is type freq and '
                        f' default is {DEF_WORD_MIN_FREQ}.'
                        f'for --level=char, characters fewer than this value'
                        f' will be excluded. default={DEF_CHAR_MIN_FREQ}')

    p.add_argument('-cv', '--char-coverage', default=DEF_CHAR_COVERAGE, type=float,
                   help='Character coverage for --level=char or --level=bpe')

    p.add_argument('-mce', '--min-co-ev', default=DEF_MIN_CO_EV, type=int,
                   help='Minimum Co-evidence for BPE merge. Valid when --level=bpe')

    p.add_argument('-spark', '--spark-master', default="local[*]", help="Spark master")
    p.add_argument('-dm', '--driver-mem', default="4g", help="Spark driver memory")
    utils.add_bool_arg(p, name='dedup', default=False,
                       help='Deduplicate the sentences: use only unique sequences')

    args = vars(p.parse_args())
    return args


def main():
    args = parse_args()
    model_path: Path = args['model']
    words_file = model_path.with_suffix(".wordfreq.gz")
    chars_file = model_path.with_suffix(".charfreq.gz")
    stats_file = chars_file if args['level'] == 'char' else words_file

    with utils.log_resources(name="extract stats"):
        master = args.pop('spark_master')
        driver_mem = args.pop('driver_mem')
        dedup = args.pop('dedup')
        if stats_file.exists():
            log.warning(f"{stats_file} exists, reusing it. please delete it if this is wrong.")
        else:
            inp = args.pop('inp')
            with spark.session(master=master, driver_mem=driver_mem) as session:
                words, chars, line_count = word_counts(paths=inp, dedup=dedup, spark=session)
            with utils.IO.writer(words_file) as out:
                write_stats(words, out, line_count=line_count)
            with utils.IO.writer(chars_file) as out:
                write_stats(chars, out, line_count=line_count)

    assert stats_file.exists()
    with utils.log_resources(name=f"learning {args['level']} vocab"):
        with utils.IO.reader(stats_file) as inp:
            learn_vocab(inp=inp, term_freqs=True, **args)


if __name__ == '__main__':
    main()
