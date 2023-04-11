import toml

from .baseparser import BaseParser


class TomlParser(BaseParser):
    format = 'toml'
    def __init__(self, config):
        super().__init__(config)

    def _config2dict(self, _config):
        return toml.load(_config)
