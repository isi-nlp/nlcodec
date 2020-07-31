#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/31/20

from pathlib import Path
from typing import Union, Tuple, Dict
import os
from contextlib import contextmanager

from nlcodec import log
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, LongType
from nlcodec.db.core import MultipartDb


def get_session(app_name="NLCodec", master=None, driver_mem=None, config: Dict[str, str] = None):
    """
    gets Spark Session
    :param app_name: name Spark App
    :param master: spark master (optional)
    :param driver_mem: driver memory (optional)
    :param config: extra config
    :return:
    """
    driver_mem = driver_mem or os.environ.get("SPARK_DRIVER_MEM", "4g")
    builder = (SparkSession.builder.appName(app_name)
               .master(master or os.environ.get("SPARK_MASTER", "local[*]"))
               .config("spark.driver.memory", driver_mem))
    if config:
        for k, v in config.items():
            builder.config(k, v)

    spark = builder.getOrCreate()
    return spark


@contextmanager
def session(*args, **kwargs):
    spark = get_session(*args, **kwargs)
    try:
        yield spark
    finally:
        spark.stop()


def read_raw_bitext(spark, src_file: Union[str, Path], tgt_file: Union[str, Path],
                    src_name='src_raw', tgt_name='tgt_raw') -> Tuple[DataFrame, int]:
    """
    reads bitext to a dataframe
    :param spark:  spark session
    :param src_file: source file to read from
    :param tgt_file:  target file to read from
    :param src_name: name for source col in DF
    :param tgt_name: name for target col in DF
    :return: DataFrame
    """
    if not isinstance(src_file, str):
        src_file = str(src_file)
    if not isinstance(tgt_file, str):
        tgt_file = str(tgt_file)

    src_df = spark.read.text(src_file).withColumnRenamed('value', src_name)
    tgt_df = spark.read.text(tgt_file).withColumnRenamed('value', tgt_name)

    n_src, n_tgt = src_df.count(), tgt_df.count()
    assert n_src == n_tgt, f'{n_src} == {n_tgt} ?'
    log.info(f"Found {n_src:,} parallel records in {src_file, tgt_file}")

    def with_idx(sdf):
        new_schema = StructType(sdf.schema.fields + [StructField("idx", LongType(), False), ])
        return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(
            schema=new_schema)

    src_df = with_idx(src_df)
    tgt_df = with_idx(tgt_df)
    bitext_df = src_df.join(tgt_df, 'idx', "inner")
    # n_bitext = bitext_df.count()
    # assert n_bitext == n_src, f'{n_bitext} == {n_src} ??'
    return bitext_df, n_src


def read_raw_bitext_tok(spark, src_path: Union[str, Path], tgt_path: Union[str, Path],
                        truncate: bool, src_len: int, tgt_len: int, src_tokenizer,
                        tgt_tokenizer) -> DataFrame:
    """
    Reads raw text and applies tokenizer in parallel
    :param spark:  SparkSession
    :param src_path: source path
    :param tgt_path: target path
    :param truncate: truncate or skip longer sequences ?
    :param src_len:  max source length
    :param tgt_len:  max target length
    :param src_tokenizer: source tokenizer
    :param tgt_tokenizer: target tokenize
    :return:  DataFrame
    """
    raw_df, n_recs = read_raw_bitext(spark, src_path, tgt_path, src_name='x', tgt_name='y')
    tok_rdd = (raw_df.rdd
               .filter(lambda r: r.x and r.y)  # exclude None
               .map(lambda r: (r.idx, src_tokenizer(r.x), tgt_tokenizer(r.y)))
               .filter(lambda r: len(r[1]) and len(r[2]))
               # exclude empty, if tokenizer created any
               )
    if truncate:
        tok_rdd = tok_rdd.map(lambda r: (r[0], r[1][:src_len], r[2][:tgt_len]))
    else:
        tok_rdd = tok_rdd.filter(lambda r: len(r[1]) < src_len and len(r[2]) < tgt_len)

    # dataframes doesnt support numpy arrays, so we cast them to python list
    a_row = tok_rdd.take(1)[0]
    if not isinstance(a_row[1], list):
        # looks like np NDArray or torch tensor
        tok_rdd = tok_rdd.map(lambda r: (r[0], r[1].tolist(), r[2].tolist()))
    tok_rdd.map(lambda r: (r[0],))
    df = tok_rdd.toDF(['id', 'x', 'y'])
    return df


def rdd_as_db(rdd, db_path: Path, field_names=('x', 'y'), repartition=0, **kwargs) -> MultipartDb:
    # input format: each rec in rdd should be : (id, (x, y)
    if repartition and repartition > 0:
        rdd = rdd.repartition(repartition)

    with MultipartDb.Writer(db_path, field_names=field_names, **kwargs) as writer:
        n = rdd.mapPartitionsWithIndex(writer).count()  # parallel write
    db = MultipartDb.load(db_path)
    log.info(f"Wrote {len(db)} recs in {len(db.part_paths)} parts at {db_path}")
    return db
