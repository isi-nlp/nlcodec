# NLCodec
A set of (low level) Natural Language Encoder-Decoders (codecs), that we generally use in preprocessing stage of 
NLP tools. The codecs include :
1. Character
2. Word
3. BPE based subword

This project contains an easy to use, and consistent python and CLI API for performing comparisons 
across the methods. 
In addition, it has a reasonably fast Byte Pair Encoding (BPE) library implemented in pure python. 
(Note: Speed comes with the cost of extra memory)


# Installation 
Please run only one of these
```bash
# Clone repo for development mode
git clone https://github.com/thammegowda/nlcodec
cd nlcodec
pip install --editable .

# Install from github
$ pip install git+https://github.com/thammegowda/nlcodec.git


# Install from pypi  (TODO: for now this code is private)
$ pip install nlcodec

```
pip installer registers a cli tool named `nlcodec` in PATH
 which serves is the command line interface. You can always trigger either via `python -m nlcodec` or 
 `python path/to/nlcodec/__main__.py` if you wish!
 

## Usage 
```bash
$ nlcodec -h
usage: nlcodec [-h] [-l {char,word,bpe}] [-i INP] [-o OUT] -m MODEL
               [-vs VOCAB_SIZE] [-idx] [-v]
               {learn,encode,decode,estimate}

positional arguments:
  {learn,encode,decode,estimate}
                        "task" or sub-command.
                            "learn" - learns vocabulary. use --level and vocab_size for type and size
                            "encode" - encodes a dataset
                            "decode" - decodes an already encoded dataset
                            "estimate" - estimates quality attributes of an encoding

optional arguments:
  -h, --help            show this help message and exit
  -l {char,word,bpe}, --level {char,word,bpe}
                        Encoding Level; Valid only for "learn" task (default:
                        None)
  -i INP, --inp INP     Input file path (default: <_io.TextIOWrapper
                        name='<stdin>' mode='r' encoding='UTF-8'>)
  -o OUT, --out OUT     Output file path. Not valid for "learn" or "estimate"
                        task (default: <_io.TextIOWrapper name='<stdout>'
                        mode='w' encoding='UTF-8'>)
  -m MODEL, --model MODEL
                        Path to model aka vocabulary file (default: None)
  -vs VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        Vocabulary size. Valid only for task=learn. (default:
                        -1)
  -idx, --indices       Indices instead of strings. Valid for task=encode and
                        task=decode (default: None)
  -v, --verbose         verbose mode. DEBUG log level. (default: False)
```

Example: 

```
# learn
head -2000 somefile.tok | nlcodec learn -l bpe -m bpe.model --vocab_size 2000

# encode  with text pieces
head  somefile.tok  | nlcodec encode -m bpe.model

# encode with indexes
head  somefile.tok  | nlcodec encode -m bpe.model -idx

# decode -- undo encoding
head  somefile.tok  | nlcodec decode -m bpe.model
head  somefile.tok  | nlcodec decode -m bpe.model -idx

# estimate quality 
head  somefile.tok  | nlcodec estimate -m bpe.model

```

# Authors 
+ [Thamme Gowda](https://twitter.com/thammegowda) 

# License
> This software is Copyright © 2019 The University of Southern California. All Rights Reserved.

Refer to [LICENSE.txt](LICENSE.txt) for full terms
