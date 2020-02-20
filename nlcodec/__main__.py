#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-29
import argparse
from typing import Dict, Any, Iterator, TextIO
import sys
from pathlib import Path

from nlcodec import learn_vocab, load_scheme, encode, decode, __version__, __description__
from nlcodec import DEF_WORD_MIN_FREQ, DEF_CHAR_MIN_FREQ, DEF_CHAR_COVERAGE


def write_lines(lines: Iterator[str], out: TextIO, line_break='\n'):
    for line in lines:
        out.write(line)
        out.write(line_break)


class MyFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def _split_lines(self, text, width: int):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(formatter_class=MyFormatter, prog='nlcodec',
                                description=__description__,
                                epilog='Visit https://github.com/isi-nlp/nlcodec for more')
    p.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    p.add_argument("task", choices=['learn', 'encode', 'decode', 'estimate'],
                   help='''R|"task" or sub-command.
    "learn" - learns vocabulary. use --level and vocab_size for type and size 
    "encode" - encodes a dataset 
    "decode" - decodes an already encoded dataset
    "estimate" - estimates quality attributes of an encoding''')

    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path. Not valid for "learn" or "estimate" task')
    p.add_argument('-m', '--model', type=Path, help='Path to model aka vocabulary file',
                   required=True)

    p.add_argument('-idx', '--indices', action='store_true', default=None,
                   help='Indices instead of strings. Valid for task=encode and task=decode')

    learn_args = p.add_argument_group("args for task=learn")
    learn_args.add_argument('-vs', '--vocab_size', type=int, default=-1,
                            help='Vocabulary size. Valid only for task=learn. This is required for'
                                 ' "bpe", but optional for "word" and "char" models, specifying it'
                                 ' will trim the vocabulary at given top most frequent types.')
    learn_args.add_argument('-l', '--level', choices=['char', 'word', 'bpe'],
                            help='Encoding Level; Valid only for task=learn')
    learn_args.add_argument('-mf', '--min_freq', default=None, type=int,
                            help='Minimum frequency of types for considering inclusion in vocabulary. '
                            'Types fewer than this frequency will be ignored. '
                            f'For --level=word or --level=bpe, freq is type freq and '
                            f' default is {DEF_WORD_MIN_FREQ}.'
                            f'for --level=char, characters fewer than this value'
                            f' will be excluded. default={DEF_CHAR_MIN_FREQ}')

    learn_args.add_argument('-cv', '--char_coverage', default=DEF_CHAR_COVERAGE, type=float,
                            help='Character coverage for --level=char or --level=bpe')

    args = vars(p.parse_args())
    return args


def main():
    args = parse_args()
    task = args.pop('task')
    if task == 'learn':
        args.pop('out')  # No output
        args.pop('indices')  # No output
        assert args.get('level'), 'argument --level is required for "learn" task'
        learn_vocab(**args)
    elif task in ('encode', 'decode'):
        scheme = load_scheme(args.pop('model'))
        inp, out, indices = args['inp'], args['out'], args.get('indices', False)
        if task == 'encode':
            recs = encode(inp, scheme, indices=indices)
            if indices:
                recs = ([str(idx) for idx in seq] for seq in recs)
            recs = (' '.join(seq) for seq in recs)
        else:
            recs = decode(inp, scheme, indices=indices)
        write_lines(recs, out)
    elif task == 'estimate':
        from nlcodec.qestim import estimate
        estimate(codec_path=args['model'], data=args['inp'])
    else:
        raise NotImplementedError(task + ' not implemented')


if __name__ == '__main__':
    main()
