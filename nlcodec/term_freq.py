#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/13/20
import json
import logging as log
from pathlib import Path
from typing import List, Dict, Tuple

from nlcodec import spark as spark_util
from nlcodec import __version__,  __epilog__

log.basicConfig(level=log.INFO)



def word_counts(paths: List[Path], dedup=True, spark=None) \
        -> Tuple[Dict[str, int], Dict[str, int], int]:
    """

    :param paths: List of paths of plain text files
    :param dedup: deduplicate lines
    :param spark: (optional) provide a spark instance;
     By default, a local spark session is launched and stopped
    :return: word_freqs, char_freqs, line_count
    """
    own_spark = not spark
    spark = spark or spark_util.get_session()
    log.info(f"Spark Session is up; you may check web UI")
    try:
        for p in paths:
            p.exists()
        dfs = [spark.read.text(str(p)) for p in paths]
        df = dfs[0]
        if len(dfs) > 1:
            for df2 in dfs[1:]:
                df = df.union(df2)
        # Deduplicate lines
        if dedup:
            log.info("Dropping duplicates")
            df = df.drop_duplicates()
        SENT_TAG = '<!S!>'
        # Tokenize
        # also include a special token, one per line to count lines
        # expand words as sep records
        word_freqs = df.rdd.flatMap(lambda row: [SENT_TAG] + row.value.strip().split()) \
            .map(lambda word: (word, 1)) \
            .reduceByKey(lambda v1, v2: v1 + v2) \
            .collectAsMap()
        line_count = word_freqs.pop(SENT_TAG)
        # word_freqs = list(sorted(word_freqs.items(), key=lambda x: x[1], reverse=True))

        # char freqs
        char_freqs = (spark.sparkContext.parallelize(word_freqs.items())
                      .flatMap(lambda r: [(ch, r[1]) for ch in r[0]])
                      .reduceByKey(lambda a, b: a + b)
                      .collectAsMap())

        return word_freqs, char_freqs, line_count

    finally:
        # only stop the own spark, not the user given spark
        if own_spark:
            log.info(f"Stopping spark session")
            spark.stop()


def write_stats(stats: Dict[str, int], out, **meta):
    stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    stats = list(stats)
    tot = sum(f for t, f in stats)

    header = dict(types=len(stats), tokens=tot)
    if meta:
        header.update(meta)
    header = json.dumps(header)
    out.write(f"#{header}\n")
    for term, freq in stats:
        out.write(f'{term}\t{freq}\n')
    log.info(f"Wrote {out}")


def main(args=None):
    args = args or parse_args()
    word_freqs, char_freqs, line_count = word_counts(args.inp, dedup=args.dedup)
    if args.word_freqs:
        write_stats(stats=word_freqs, out=args.word_freqs, line_count=line_count)
    if args.char_freqs:
        write_stats(stats=char_freqs, out=args.char_freqs, line_count=line_count)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                epilog=__epilog__)
    p.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    p.add_argument('-i', '--inp', type=Path, nargs='+',
                   help='Input file paths')
    p.add_argument('-wf', '--word_freqs', type=argparse.FileType('w'), default=None,
                   help='Output file path for word frequencies')
    p.add_argument('-cf', '--char_freqs', type=argparse.FileType('w'), default=None,
                   help='Output file path for character frequencies')

    p.add_argument('-dd', '--dedup', action='store_true',
                   default=True, help='Deduplicate the sentences: use only unique sequences')
    p.add_argument('-ndd', '--no-dedup', dest='dedup', action='store_false', default=False,
                   help='Do not deduplicate. ')
    args = p.parse_args()
    assert args.word_freqs or args.char_freqs, \
        'At least one of --word_freqs --char_freqs paths be given to write the stats'
    return args


if __name__ == '__main__':
    main()
