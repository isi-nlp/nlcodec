#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25

__version__ = '0.4.0'
__description__ = """nlcodec is a collection of encoding schemes for natural language sequences. 
nlcodec.db is a efficient storage and retrieval layer for integer sequences of varying lengths."""
PROJECT_HOME = 'https://github.com/isi-nlp/nlcodec'

from pathlib import Path

__epilog__ = f'Visit https://github.com/isi-nlp/nlcodec or https://isi-nlp.github.io/nlcodec/' \
             f' to learn more. You\'re currently using version {__version__} loaded ' \
             f' from {Path(__file__).parent}'

DEF_MIN_CO_EV = 95  # recommended by Gowda and May (2020)
DEF_WORD_MIN_FREQ = 1  # minimum times a word should exist to be used for word vocab
DEF_CHAR_MIN_FREQ = 20  # minimum times a char should be seen to be included in init vocab
DEF_CHAR_COVERAGE = 0.9995  # Credits to google/sentencepiece for this idea;

import logging
log = logging.getLogger("nlcodec")
logging.basicConfig(
    level=logging.INFO, datefmt='%m-%d %H:%M:%S',
    format='[%(asctime)s] p%(process)s {%(module)s:%(lineno)d} %(levelname)s - %(message)s')

from nlcodec.codec import (EncoderScheme, WordScheme, CharScheme, BPEScheme, Type, Reseved,
                           REGISTRY,
                           learn_vocab, load_scheme, Level, encode, decode)
from nlcodec.dstruct import LnNode, TrNode, MaxHeap
