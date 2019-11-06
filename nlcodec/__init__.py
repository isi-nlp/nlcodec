#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25

__version__ = 0.1
__description__ = """nlcodec is a collection of encoder-decoder schemes for natural language text"""

import logging as log
log.basicConfig(level=log.INFO)

from nlcodec.codec import (EncoderScheme, WordScheme, CharScheme, BPEScheme, Type, Reseved, REGISTRY,
                    learn_vocab, load_scheme, Level, encode, decode)
from nlcodec.dstruct import LnNode, TrNode, MaxHeap
