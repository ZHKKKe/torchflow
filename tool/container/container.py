

class Container:
    def __init__(self):
        self._dict = {}

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return self._dict.__contains__(key)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if callable(value) and key is None:
            key = value.__name__

        if key is not None:       
            if not key in self._dict:
                self._create_item(key, value)
            else:
                self._update_item(key, value)

    def __delitem__(self, key):
        if key in self._dict.keys():
            del self._dict[key]

    def _create_item(self, key, value):
        self._dict[key] = value

    def _update_item(self, key, value):
        self._dict[key] = value

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()
