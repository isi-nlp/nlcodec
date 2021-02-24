#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/31/20

import logging as log
from pathlib import Path
from nlcodec import load_scheme
from nlcodec import spark as spark_util
from nlcodec import utils

from nlcodec import __version__,  __epilog__

log.basicConfig(level=log.INFO)


def main(args=None):
    args = args or parse_args()
    src_codec = load_scheme(args.src_model)
    tgt_codec = src_codec
    if args.tgt_model:
        tgt_codec = load_scheme(args.tgt_model)

    with spark_util.session() as session:
        df = spark_util.read_raw_bitext_tok(
            spark=session, src_path=args.src_text, tgt_path=args.tgt_text, truncate=args.truncate,
            src_len=args.src_len, tgt_len=args.tgt_len,
            src_tokenizer=src_codec.encode, tgt_tokenizer=tgt_codec.encode)
        rdd = df.rdd.map(lambda r: (r.id, (r.x, r.y)))
        db = spark_util.rdd_as_db(rdd=rdd, db_path=args.db_path, field_names=('x', 'y'),
                                  repartition=args.num_parts)
        log.info(f"stored {len(db)} recs at {args.db_path}")


# noinspection PyTypeChecker
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Encodes bitext nlcdec and stores them in db, "
                                            " using pyspark backend.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                epilog=__epilog__)
    p.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    p.add_argument("-sm", "--src-model", metavar='PATH', type=Path, required=True, help="source model")
    p.add_argument("-tm", "--tgt-model", metavar='PATH',type=Path, required=False,
                   help="target model; default=same as src-model")
    p.add_argument("-st", "--src-text", metavar='PATH', type=Path, required=True, help="source text")
    p.add_argument("-tt", "--tgt-text", metavar='PATH', type=Path, required=True, help="target text")

    p.add_argument("-db", "--db", metavar='PATH', dest='db_path', type=Path, required=True,
                   help="Path to store db")
    p.add_argument("-np", "--num-parts", metavar='N', type=int, default=0,
                   help="A value greater than 0 forces the db to have these many parts")

    p.add_argument("-sn", "--src-col", type=str, default='x', help="source column name")
    p.add_argument("-tn", "--tgt-col", type=str, default='y', help="target column name")

    p.add_argument("-sl", "--src-len", type=int, default=256, help="source length max")
    p.add_argument("-tl", "--tgt-len", type=int, default=256, help="target length max")
    utils.add_bool_arg(p, 'truncate', default=True, help='truncate longer sentences',
                       nhelp="drop longer sentences")

    p.add_argument("-spark", "--spark_master", default='local[*]', help="Use Spark master")
    p.add_argument("-dm", "--driver_mem", default='4g', help="Memory for spark driver")

    args = p.parse_args()
    assert args.src_text.exists()
    assert args.tgt_text.exists()
    return args


if __name__ == '__main__':
    main()
