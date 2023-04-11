import argparse


class Namespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getattr__(self, item):
        return None


class BaseParser:
    def __init__(self, config):
        self._config = config
        self._dict = self._config2dict(self._config)
        self._args = Namespace(**self._dict2args(self._dict))

    def parse_args(self):
        return self._args
    
    def _dict2args(self, _dict):
        _args = Namespace()
        _args_dict = vars(_args)

        for key in _dict:
            value = _dict[key]
            if isinstance(value, dict):
                value = Namespace(**self._dict2args(value))
            _args_dict[key] = value
        
        return _args_dict

    def _config2dict(self, _config):
        raise NotImplementedError
