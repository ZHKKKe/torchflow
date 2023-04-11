import argparse
from .tomlparser import TomlParser
from .dictparser import DictParser

from torchflow.tool import logger


PARSERS = [
    TomlParser,
    DictParser,
]


def parse_args(config):
    # TODO: support `config` in the Namespace format

    try:
        suffix = config.split('.')[-1]
    except:
        if isinstance(config, dict):
            suffix = 'dict'

    args = None
    for parser_cls in PARSERS:
        if suffix == parser_cls.format:
            _parser = parser_cls(config)
            args = _parser.parse_args()
            break
    return args


def print_args(args, depth=1):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    for key in args:
        value = args[key]
        if isinstance(value, argparse.Namespace):
            logger.log('{0}--{1}: '.format('  ' * depth, key))
            print_args(value, depth=depth+1)
        elif isinstance(value, (list, tuple)):
            logger.log('{0}--{1}: ('.format('  ' * depth, key))
            for element in value:
                logger.log('{0}{1},'.format('  ' * (depth+1), element))
            logger.log('{0})'.format('  ' * depth))
        else:
            logger.log('{0}--{1}: {2}'.format('  ' * depth, key, value))


def fetch_arg(arg, default, unassigned=[None, '']):
    return default if arg in unassigned else arg
