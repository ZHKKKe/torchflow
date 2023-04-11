from .baseparser import BaseParser


class DictParser(BaseParser):
    format = 'dict'
    def __init__(self, config):
        super().__init__(config)

    def _config2dict(self, _config):
        return _config
