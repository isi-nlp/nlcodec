# BPE++
Byte Pair Encoding ++ is a fast Byte Pair Encoding library implemented in pure python.
It supports Hierarchical BPE of two levels: 
1. Recursively Merge character pairs into sub words
2. Recursively Merge subwords into phrases


# Installation 
```bash
# Install from pypy 
$ pip install bpepp

# Install from github
$ pip install git+https://github.com/thammegowda/bpepp.git
```

## Usage 
```bash
$ bpepp -h
usage: bpepp [-h] [-v] {learn1,learn2,learn,encode,decode} ...

Byte Pair Encoding++ is a pure python library which implements Hierarchical
merging of character streams into words and phrases.

positional arguments:
  {learn1,learn2,learn,encode,decode}
    learn1              Learn level1 BPE
    learn2              Learn level2 BPE, from level1
    learn               learn1 and learn2
    encode              Encode seqs
    decode              Decode seqs

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
```



# Authors 
+ [Thamme Gowda](https://twitter.com/thammegowda) 

# License 
> This software is Copyright © 2019 The University of Southern California. All Rights Reserved.

Refer to [LICENSE.txt](LICENSE.txt) for full terms