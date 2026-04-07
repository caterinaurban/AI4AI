from copy import deepcopy
from typing import Dict, Tuple

from abstract_domains.abstract_domain import AbstractDomain, colors
from frontend.mirror import NAP


class DeepPolyDomain(AbstractDomain):

    def __init__(self, ranges: Dict[str, Tuple[float, float]]):
        super().__init__(ranges)
        self._poly = dict()
        for ipt, val in ranges.items():
            lower = dict()
            lower['_'] = val[0]
            upper = dict()
            upper['_'] = val[1]
            self._poly[ipt] = (lower, upper)
        self._polarities = dict()

    @property
    def poly(self):
        return self._poly

    @property
    def polarities(self):
        return self._polarities

    def substitute(self, dictionary: Dict[str, float]):
        def is_input(var):
            return self.colors[var] == colors['input']
        inf = deepcopy(dictionary)
        while any(var in inf and not is_input(var) for var in self.poly):
            for var in self.poly:
                if var in inf and not is_input(var):
                    coeff = inf.pop(var)
                    if coeff > 0:
                        replacement = self.poly[var][0]
                    elif coeff < 0:
                        replacement = self.poly[var][1]
                    else:
                        replacement = {'_': 0.0}
                    for var, val in replacement.items():
                        inf[var] = inf.get(var, 0) + coeff * val
        sup = deepcopy(dictionary)
        while any(var in sup and not is_input(var) for var in self.poly):
            for var in self.poly:
                if var in sup and not is_input(var):
                    coeff = sup.pop(var)
                    if coeff > 0:
                        replacement = self.poly[var][1]
                    elif coeff < 0:
                        replacement = self.poly[var][0]
                    else:
                        replacement = {'_': 0.0}
                    for var, val in replacement.items():
                        sup[var] = sup.get(var, 0) + coeff * val
        return inf, sup

    def evaluate(self, dictionary: Dict[str, float]):
        inf, sup = self.substitute(dictionary)
        lower = super().evaluate(inf)[0]
        upper = super().evaluate(sup)[1]
        return lower, upper

    def affine(self, layer):
        if self.is_bottom():
            return self
        for lhs, rhs in layer.items():
            self.poly[lhs] = (deepcopy(rhs), deepcopy(rhs))
            lower, upper = self.evaluate(rhs)
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
                self.poly[lhs] = ({'_': 0.0}, {'_': 0.0})
                self.flags[lhs] = -1
            elif 0 <= lower or (nap and nap.is_active(lhs)):
                if (nap and nap.is_active(lhs)) and lower < 0:
                    self.bounds[lhs] = (0, upper)
                    self.poly[lhs] = ({'_': 0.0}, self.poly[lhs][1])
                self.flags[lhs] = 1
            else:
                assert lower < 0 < upper
                self.polarities[lhs] = abs((lower + upper) / (upper - lower))
                if upper <= -lower:     # case (b) in Fig. 4
                    self.bounds[lhs] = (0, upper)
                    inf = {'_': 0.0}
                else:   # case (c) in Fig. 4
                    self.bounds[lhs] = (lower, upper)
                    inf = deepcopy(self.poly[lhs][0])
                m = upper / (upper - lower)
                if m > 0:
                    sup = deepcopy(self.poly[lhs][1])
                elif m < 0:
                    sup = deepcopy(self.poly[lhs][0])
                else:  # m == 0
                    sup = {'_': 0.0}
                for var, val in sup.items():
                    sup[var] = m * val
                q = - upper * lower / (upper - lower)
                sup['_'] = sup['_'] + q
                self.poly[lhs] = (inf, sup)
                self.flags[lhs] = 0
            self.colors[lhs] = colors['hidden']
        return self
