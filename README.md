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
  (other implementations require c++ expertise to modify the core) 
2. BPE model is a simple text that can be inspected with `less` or `cut`. It includes info on which pieces were put together and what frequencies etc. 
3. Reasonably faster than the other pure python implementations -- speed in python comes with the cost of extra memory due to indexing.
4. PySpark backend for extracting term frequencies from large datasets 


# Installation 
Please run only one of these
```bash
# Clone repo for development mode (preferred  mode)
git clone https://github.com/isi-nlp/nlcodec
cd nlcodec
pip install --editable . 

# Install from github, directly
$ pip install git+https://github.com/isi-nlp/nlcodec.git


# Install from pypi
$ pip install nlcodec
```
pip installer registers a cli tool named `nlcodec` in PATH
 which serves is the command line interface.
  You can always trigger either via `python -m nlcodec` or 
 `python path/to/nlcodec/__main__.py` if you wish!

Docs are available at  
- locally at [docs/intro.adoc](docs/intro.adoc)
- HTML format: https://isi-nlp.github.io/nlcodec


# Authors 
+ [Thamme Gowda](https://twitter.com/thammegowda) 
