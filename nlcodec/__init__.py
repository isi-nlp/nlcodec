#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25

__version__ = '0.2.0'
__description__ = """nlcodec is a collection of encoding schemes for natural language sequences"""

DEF_MIN_CO_EV = 5
DEF_WORD_MIN_FREQ = 1  # minimum times a word should exist to be used for word vocab
DEF_CHAR_MIN_FREQ = 20  # minimum times a char should be seen to be included in init vocab
DEF_CHAR_COVERAGE = 0.9995  # Credits to google/sentencepiece for this idea;


import logging as log
log.basicConfig(level=log.INFO)

from nlcodec.codec import (EncoderScheme, WordScheme, CharScheme, BPEScheme, Type, Reseved, REGISTRY,
                    learn_vocab, load_scheme, Level, encode, decode)
from nlcodec.dstruct import LnNode, TrNode, MaxHeap
