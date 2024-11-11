from numbers import Number

class Accumulator(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def add(self, value):
        if isinstance(value, Number):
            self.sum += value
            self.count += 1
        else:
            self.sum += value.sum
            self.count += value.count
    
    def reset(self):
        self.sum = 0
        self.count = 0
    
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0
    
    def __len__(self):
        return self.count
    
    def __add__(self, value):
        tmp = Accumulator()
        tmp.add(self)
        tmp.add(value)
        return tmp
    
    def __iadd__(self, value):
        self.add(value)
        return self

    def __str__(self):
        return f'{self.avg:.2f}'
    
class Accumulators:
    def __init__(self):
        self.accs = None

    def update(self, values):
        if self.accs is None:
            self.accs = {key: Accumulator() for key in values.keys()}
        for key, value in values.items():
            self.accs[key].add(value)

    def reset(self):
        self.accs = None
    
    def items(self):
        return {key: acc.avg for key, acc in self.accs.items()} if self.accs else {}

    def __str__(self):
        desc_str = ", ".join([f"{key}: {acc.avg:.6f}" for key, acc in self.accs.items()]) if self.accs else "None"
        return desc_str