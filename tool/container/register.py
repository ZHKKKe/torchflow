from .container import Container


class Register(Container):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def register(self, target):
        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            return add(None, target)
        return lambda x: add(target, x)
