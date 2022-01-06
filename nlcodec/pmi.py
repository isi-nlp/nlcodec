from typing import List, TextIO, Dict, Tuple, Union, Iterator, Optional, Set
import multiprocessing as mp
from nlcodec import __version__, log
import functools as fn
import math

class PMIFuncs():

    # Most of the methods use 'Type' from codec.py but removed them from typing to prevent circular dependancy.

    space_tok = '▁' # Referring Reseved would have created a circular dependency
    ngram_variants = ['naive_pmi', 'avg_pmi', 'min_pmi']
    sgram_variants = ['skip_pmi']

    @classmethod
    def get_pmis(cls, table:List['Type'], nterms:int,
            nlines:int, bigram_freqs:Dict[str,int]=None, 
            pmi_variant:str='naive_pmi') -> List[Tuple['Type',float]]:
        sorter_func = cls.get_pmi_func(pmi_variant)
        table_pmis = []
        for token in table:
            pmi = sorter_func(token, nterms, nlines, bigram_freqs)
            table_pmis.append((token, pmi))
        return table_pmis

    @classmethod
    def get_pmi_func(cls, variant:str):
        if variant == 'naive_pmi':
            return cls.naive_pmi
        elif variant == 'avg_pmi':
            return cls.avg_pmi
        elif variant == 'min_pmi':
            return cls.min_pmi
        elif variant == 'skip_pmi':
            return cls.skip_pmi
        return ValueError(f'Variant {variant} not available. \
                            Options : naive_pmi, avg_pmi, min_pmi')

    @classmethod
    def skip_pmi(cls, tok:'Type', nterms:int, nlines:int, *args) -> float:
        ngram = len(tok.kids)
        word_probs = [k/nterms for k in tok.kids if k > 0]
        sgram_prob = tok.freq / (nterms - (nlines*(ngram-1)))
        return cls._naive_pmi(sgram_prob, word_probs)

    @classmethod
    def naive_pmi(cls, tok:'Type', nterms:int, nlines:int, *args) -> float:
        ngram = len(tok.kids)
        word_probs = [ k/nterms for k in tok.kids ]
        ngram_prob = tok.freq / (nterms - (nlines*(ngram-1)))
        return cls._naive_pmi(ngram_prob, word_probs)

    @classmethod
    def avg_pmi(cls, tok:'Type', nterms:int, nlines:int, 
                bigram_freqs:Dict[str,int]) -> float:
        bigram_probs = { name : freq/(nterms-nlines) for name,freq in bigram_freqs.items()}
        pmis_list = cls._get_bigram_pmis(tok, nterms, bigram_probs)
        return fn.reduce(lambda a,b: a+b, pmis_list) / len(pmis_list)

    @classmethod
    def min_pmi(cls, tok:'Type', nterms:int, nlines:int,
                bigram_freqs:Dict[str,int]) -> float:
        bigram_probs = { name : freq/(nterms-nlines) for name,freq in bigram_freqs.items()}
        pmis_list = cls._get_bigram_pmis(tok, nterms, bigram_probs)
        return min(pmis_list)

    @staticmethod
    def _naive_pmi(ngram_prob:float, word_probs:List[float]) -> float:
        pmi_num = ngram_prob
        pmi_dec = fn.reduce(lambda a,b: a*b, word_probs)
        return math.log(pmi_num / pmi_dec)

    @classmethod
    def _get_bigram_pmis(cls, token:'Type', nterms:int, 
                        bigram_probs:Dict[str,float]) -> List[float]:
        parts = token.name.replace(cls.space_tok, f'{cls.space_tok} ').split()[:-1]
        bigrams = [''.join(parts[i:i+2]) for i in range(len(parts)-1)]
        word_probs = [ k/nterms for k in token.kids]
        pmis_list = []
        for x, bigram in enumerate(bigrams):
            prob = bigram_probs[bigram]
            pmi = cls._naive_pmi(prob, word_probs[x:x+2])
            pmis_list.append(pmi)
        return pmis_list

