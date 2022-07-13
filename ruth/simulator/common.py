from itertools import product


class ParamsGenerator:
    def __init__(self):
        self.params = dict()

    def register(self, name, iterable):
        assert name not in self.params, f"Parameter '{name}' is already registered."
        self.params[name] = iterable

    def drop(self, name):
        del self.params[name]

    def __iter__(self):
        prod = product(*self.params.values())
        keys = self.params.keys()

        for comb in prod:
            yield dict(zip(keys, comb))

    def __len__(self):
        return len(list(product(*self.params.values())))
