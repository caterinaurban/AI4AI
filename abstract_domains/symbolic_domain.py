from copy import deepcopy
from typing import Dict, Tuple

from abstract_domains.abstract_domain import AbstractDomain, colors
from frontend.mirror import NAP


class SymbolicDomain(AbstractDomain):

    def __init__(self, ranges: Dict[str, Tuple[float, float]]):
        super().__init__(ranges)
        self._symbols = dict()

    @property
    def symbols(self):
        return self._symbols

    def substitute(self, dictionary: Dict[str, float]):
        _dictionary = deepcopy(dictionary)
        for sym, exp in self.symbols.items():
            coeff = _dictionary.get(sym, 0)
            _dictionary.pop(sym, None)
            for var, val in exp.items():
                if var in _dictionary:
                    _dictionary[var] += coeff * val
                else:
                    _dictionary[var] = coeff * val
        return _dictionary

    def evaluate(self, dictionary: Dict[str, float]):
        _rhs = self.substitute(dictionary)
        lower, upper = super().evaluate(_rhs)
        return lower, upper

    def affine(self, layer):
        if self.is_bottom():
            return self
        for lhs, rhs in layer.items():
            self.symbols[lhs] = self.substitute(rhs)
            (lower, upper) = self.evaluate(rhs)
            if 0 < lower - upper < 1e-8:
                print("ROUNDING", lhs, lower, upper)
                value = round((lower + upper) / 2, 7)
                lower = upper = value
            self.bounds[lhs] = (lower, upper)
            self.colors[lhs] = colors['output']
        return self

    def relu(self, layer, nap: NAP = None):
        if self.is_bottom():
            return self
        # bounds, symbols, expressions, polarities, flags = deepcopy(symbolic)
        for lhs in layer:
            lower, upper = self.bounds[lhs]
            if upper <= 0 or (nap and nap.is_inactive(lhs)):
                self.bounds[lhs] = (0, 0)
                zero: Dict[str, float] = dict()
                zero['_'] = 0.0
                self.symbols[lhs] = zero
                self.flags[lhs] = -1
            elif 0 <= lower or (nap and nap.is_active(lhs)):
                if (nap and nap.is_active(lhs)) and lower < 0:
                    self.bounds[lhs] = (0, upper)
                    del self.symbols[lhs]
                self.flags[lhs] = 1
            else:
                assert lower < 0 < upper
                self.polarities[lhs] = abs((lower + upper) / (upper - lower))
                self.bounds[lhs] = (0, upper)
                del self.symbols[lhs]
                self.flags[lhs] = 0
            self.colors[lhs] = colors['hidden']
        return self

    def print(self, label=None):
        super().print(label)
        print('symbols: ')
        for sym, exp in self.symbols.items():
            symbol = ' + '.join('%.2f * %s' % (coeff, var) for var,coeff in exp.items())
            print('{} -> {}'.format(sym, symbol))
