import math
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Dict, Tuple

from colorama import Fore, Style

from frontend.mirror import NAP


class Abstraction(Enum):
    INTERVALS = 0
    SYMBOLIC = 1
    DEEPPOLY = 2
    DEEPPOLY0 = 3   # only the first ReLU abstraction
    NEURIFY = 6
    PRODUCT = 7     # symbolic + deeppoly


colors = {
    'input': Fore.YELLOW,
    'hidden': Fore.LIGHTBLACK_EX,
    'output': Fore.MAGENTA
}


class AbstractDomain:

    def __init__(self, ranges: Dict[str, Tuple[float, float]]):
        self._colors = dict()
        self._bounds: Dict[str, Tuple[float, float]] = dict()
        for ipt, val in ranges.items():
            self._bounds[ipt] = val
            self._colors[ipt] = colors['input']
        self._flags = dict()
        self._polarities = dict()

    @property
    def bounds(self):
        return self._bounds

    @property
    def flags(self):
        return self._flags

    @property
    def polarities(self):
        return self._polarities

    @property
    def colors(self):
        return self._colors

    def is_bottom(self):
        for var, val in self.bounds.items():
            if val[0] > val[1]:
                print(var, val[0], val[1])
        return any(val[0] > val[1] for val in self.bounds.values())

    def evaluate(self, dictionary: Dict[str, float]):
        result = (0, 0)
        for var, coeff in dictionary.items():
            if var != '_':
                a = coeff * self.bounds[var][0]
                b = coeff * self.bounds[var][1]
                result = (result[0] + min(a, b), result[1] + max(a, b))
            else:
                result = (result[0] + coeff, result[1] + coeff)
        return result

    def affine(self, layer):
        if self.is_bottom():
            return self
        for lhs, rhs in layer.items():
            _rhs = deepcopy(rhs)
            (lower, upper) = self.evaluate(_rhs)
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
        for lhs in layer:
            lower, upper = self.bounds[lhs]
            if upper <= 0 or (nap and nap.is_inactive(lhs)):
                self.bounds[lhs] = (0, 0)
                self.flags[lhs] = -1
            elif 0 <= lower or (nap and nap.is_active(lhs)):
                if (nap and nap.is_active(lhs)) and lower < 0:
                    self.bounds[lhs] = (0, upper)
                self.flags[lhs] = 1
            else:
                assert lower < 0 < upper
                self.polarities[lhs] = abs((lower + upper) / (upper - lower))
                self.bounds[lhs] = (0, upper)
                self.flags[lhs] = 0
            self.colors[lhs] = colors['hidden']
        return self

    def sigmoid(self, layer):
        if self.is_bottom():
            return self
        for lhs in layer:
            lower, upper = self.bounds[lhs]
            _lower = 1 / (1 + math.exp(-lower))
            _upper = 1 / (1 + math.exp(-upper))
            self.bounds[lhs] = (_lower, _upper)
            if _upper <= 0:
                self.flags[lhs] = -1
            elif 0 <= _lower:
                self.flags[lhs] = 1
            else:
                self.flags[lhs] = 0
            self.colors[lhs] = colors['hidden']
        return self

    def outcome(self, outputs, log=False):
        if self.is_bottom():
            return '⊥'
        else:
            for outA in outputs:
                found = True
                for outB in outputs:
                    if outA != outB:
                        lower, upper = self.evaluate({outA: 1, outB: -1})
                        if log:
                            diff = '{} - {}'.format(outA, outB)
                            print('{}: [{}, {}]'.format(diff, lower, upper))
                        if lower < 0:
                            found = False
                if found:
                    return "{}".format(outA)
            return "?"

    def print(self, label=None, subset=None):
        if label:
            print(label)
        for var, val in self.bounds.items():
            if subset is None or var in subset:
                binding = "%s: [%.2f, %.2f]" % (var, val[0], val[1])
                print(self.colors[var] + binding, Style.RESET_ALL)
