# NLCodec

NOTE: The docs are available at https://isi-nlp.github.io/nlcodec

A set of (low level) Natural Language Encoder-Decoders (codecs), that are useful in preprocessing stage of 
NLP pipeline. These codecs include encoding of sequences into one of the following:
1. Character
2. Word
3. BPE based subword

It provides python (so embed into your app) and CLI APIs (use it as stand alone tool).

There are many BPE implementations available already, but this one provides differs:
1. Pure python implementation that is easy to modify anything to try new ideas. 
  (other implementations require c++/rust expertise to modify the core) 
2. An easily shareable and inspectable model file. It is a simple text that can be inspected with `less` or `cut`. It includes info on which pieces were put together and what frequencies etc. 
3. Reasonably faster than the other pure python implementations. Under the hood  tries, doubly linked lists, max-heaps, hash maps etc data-structures to boost performance.
4. PySpark backend for extracting term frequencies from large datasets. 

# Installation 
Please run only one of these
```bash
# Install from pypi (preferred)
$ pip install nlcodec --ignore-installed 

# Clone repo for development mode 
git clone https://github.com/isi-nlp/nlcodec
cd nlcodec
pip install --editable . 

```
pip installer registers these CLI tools in your PATH: 
+ `nlcodec`  -- CLI  for learn, encode, decode. Same as `python -m nlcodec`
+ `nlcodec-learn`  -- CLI  for learn BPE, with PySpark backend. Same as `python -m nlcodec.learn`  
+ `nlcodec-db` -- CLI for bitextdb. `python -m nlcodec.bitextdb`
+ `nlcodec-freq` -- CLI for extracting word and char frequencies using spark backend. 
 

Docs are available at  
- HTML format: https://isi-nlp.github.io/nlcodec (recommended)
- Locally at [docs/intro.adoc](docs/intro.adoc)


### Citation
Refer to https://arxiv.org/abs/2104.00290

```bibtex
@misc{gowda2021manytoenglish,
      title={Many-to-English Machine Translation Tools, Data, and Pretrained Models},
      author={Thamme Gowda and Zhao Zhang and Chris A Mattmann and Jonathan May},
      year={2021},
      eprint={2104.00290},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Authors 
+ [Thamme Gowda](https://twitter.com/thammegowda) 
