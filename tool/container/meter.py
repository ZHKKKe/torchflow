from . import container


class MeanMeterItem:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.value = value
        self.sum += value
        self.count += 1
        self.mean = self.sum / self.count

    def __format__(self, format):
        return "{self.value:{format}} "\
               "({self.mean:{format}})".format(self=self, format=format)


class MeanMeter(container.Container):
    def __init__(self):
        super().__init__()

    def _create_item(self, key, value):
        self._dict[key] = MeanMeterItem()
        self._update_item(key, value)
    
    def _update_item(self, key, value):
        self._dict[key].update(value)

    def reset(self, key=None):
        if key is None:
            for meter in self._dict.values():
                meter.reset()
        elif key in self._dict.keys():
            self._dict[key].reset()
