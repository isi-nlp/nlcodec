# Changelog

## 0.3.0

- add `nlcodec-freqs` CLI to setup.py
- log time and memory usage for `learn` task
- log BPE merge operations once every 2s instead of all operations
- using`__slots__`: ~25% faster, %30 less memory for BPE with 3M word types
- `nlcodec.db.core` with `Db` and `MultipartDb` 
- `nlcodec.db.batch` with `Batch` and `BathIterable` 
- CLI `nlcodec.learn` for learning BPE using pyspark
- CLI `nlcodec.bitextdb`  to build a database from parallel text


## 0.2.4 : 2020-07-14
- fix issue with `name` as class property (#24, #25)


## 0.2.3 : 2020-07-07
- Option to supply preconfigured `spark` session object
- Add docs 


## 0.2.2 : 2020-06-14
- Option to accept term frequencies as input
- PySpark backend to compute word and char frequencies
- `--min-co-ev` of BPE is CLI arg

## 0.2.1 : 2020-05-30
- FIX: `find_packages()` in `setup.py` file to include nested packages

## 0.2.0 : 2020-04-17
- uploaded to pypi : `pip install nlcodec`
- public repository with apache license 2.0

