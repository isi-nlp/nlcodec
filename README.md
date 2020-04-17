# NLCodec
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
 

## Usage 
```bash
$ python -m nlcodec -h
usage: __main__.py [-h] [-i INP] [-o OUT] -m MODEL [-idx] [-vs VOCAB_SIZE]
                   [-l {char,word,bpe}] [-mf MIN_FREQ]
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
  -i INP, --inp INP     Input file path (default: <_io.TextIOWrapper
                        name='<stdin>' mode='r' encoding='UTF-8'>)
  -o OUT, --out OUT     Output file path. Not valid for "learn" or "estimate"
                        task (default: <_io.TextIOWrapper name='<stdout>'
                        mode='w' encoding='UTF-8'>)
  -m MODEL, --model MODEL
                        Path to model aka vocabulary file (default: None)
  -idx, --indices       Indices instead of strings. Valid for task=encode and
                        task=decode (default: None)

args for task=learn:
  -vs VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        Vocabulary size. Valid only for task=learn. This is
                        required for "bpe", but optional for "word" and "char"
                        models, specifying it will trim the vocabulary at
                        given top most frequent types. (default: -1)
  -l {char,word,bpe}, --level {char,word,bpe}
                        Encoding Level; Valid only for task=learn (default:
                        None)
  -mf MIN_FREQ, --min_freq MIN_FREQ
                        Minimum frequency of types for considering inclusion
                        in vocabulary. Types fewer than this frequency will be
                        ignored. For --level=word, freq is type freq and
                        default is 2.for --level=char or --level=bpe,
                        characters fewer than this value will be excluded.
                        default=20 (default: None)

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

## Python API

### Using a vocabulary
```python
from nlcodec import  load_scheme
path = 'path/to/vocab.model'
vocab = load_scheme(path)

line = 'this is a sample sentence'
# encode a line of text into list of ids
vocab.encode(line)

# parallel encode a bunch of lines using multiple cpus
vocab.encode_parallel(seqs=[line], n_cpus=2)

# encode a line of text into pieces 
vocab.encode_str(line)

# decode
vocab.decode(vocab.encode(line))
vocab.decode_str(vocab.encode_str(line))
```

### Creating a vocabulary
```python
from nlcodec import learn_vocab
inp = ['line 1', 'line 2']
level = 'bpe' # other options = char, word
model = 'path/to/vocab.model'
learn_vocab(inp, level, model, vocab_size=8000, min_freq=1, char_coverage=0.9995)
```


### BPE Subword sub optimal splits for regularization
 
```python
from nlcodec import load_scheme, BPEScheme
path = 'path/to/bpe-vocab.model'
bpe: BPEScheme = load_scheme(path)
some_type = bpe.table[1000] # select some bpe piece type

# get stochastic split
some_type.get_stochastic_split(split_ratio=0.5, name=False)
# get all possible permutations 
some_type.get_permutations(name=False)

```


# Authors 
+ [Thamme Gowda](https://twitter.com/thammegowda) 
