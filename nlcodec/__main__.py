#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-29
import argparse
from typing import Dict, Any, Iterator, TextIO
import sys
from pathlib import Path

from nlcodec import log, learn_vocab, load_scheme, encode, decode


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
    p = argparse.ArgumentParser(formatter_class=MyFormatter)
    p.add_argument("task", choices=['learn', 'encode', 'decode', 'estimate'],
                   help='''R|"task" or sub-command.
    "learn" - learns vocabulary. use --level and vocab_size for type and size 
    "encode" - encodes a dataset 
    "decode" - decodes an already encoded dataset
    "estimate" - estimates quality attributes of an encoding''')

    p.add_argument('-l', '--level', choices=['char', 'word', 'bpe'],
                   help='Encoding Level; Valid only for "learn" task')

    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path. Not valid for "learn" or "estimate" task')
    p.add_argument('-m', '--model', type=Path, help='Path to model aka vocabulary file',
                   required=True)
    p.add_argument('-vs', '--vocab_size', type=int, default=-1,
                   help='Vocabulary size. Valid only for task=learn.')
    p.add_argument('-idx', '--indices', action='store_true', default=None,
                   help='Indices instead of strings. Valid for task=encode and task=decode')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose mode. DEBUG log level.')
    args = vars(p.parse_args())
    if args.pop('verbose'):
        log.getLogger().setLevel(level=log.DEBUG)
    return args


def main():
    args = parse_args()
    task = args.pop('task')
    if task == 'learn':
        args.pop('out')      # No output
        args.pop('indices')  # No output
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
