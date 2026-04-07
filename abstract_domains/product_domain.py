import math
from math import inf
from typing import List, Type

from abstract_domains.abstract_domain import AbstractDomain, colors
from abstract_domains.deeppoly_domain import DeepPolyDomain
from abstract_domains.symbolic_domain import SymbolicDomain
from frontend.mirror import NAP


class ProductDomain(AbstractDomain):

    def __init__(self, ranges, domains: List[Type[AbstractDomain]]):
        super().__init__(ranges)
        self._domains = list()
        for domain in domains:
            self._domains.append(domain(ranges))

    @property
    def domains(self):
        return self._domains

    def reduce(self, variable):
        lower = max(domain.bounds[variable][0] for domain in self.domains)
        upper = min(domain.bounds[variable][1] for domain in self.domains)
        if abs(upper - lower) < 1e-5:
            midpoint = (lower + upper) / 2
            lower = upper = midpoint
        self.bounds[variable] = (lower, upper)
        for i, domain in enumerate(self.domains):
            domain.bounds[variable] = (lower, upper)
            self.domains[i] = domain

    def affine(self, layer):
        if self.is_bottom():
            return self
        for i, domain in enumerate(self.domains):
            self.domains[i] = domain.affine(layer)
        for lhs in layer.keys():
            self.reduce(lhs)
            self.colors[lhs] = colors['output']
        return self

    def relu(self, layer, nap: NAP = None):
        if self.is_bottom():
            return self
        for lhs in layer:
            lower, upper = self.bounds[lhs]
            if upper <= 0 or (nap and nap.is_inactive(lhs)):
                self.flags[lhs] = -1
            elif 0 <= lower or (nap and nap.is_active(lhs)):
                self.flags[lhs] = 1
            else:
                assert lower < 0 < upper
                self.polarities[lhs] = abs((lower + upper) / (upper - lower))
                self.flags[lhs] = 0
        for i, domain in enumerate(self.domains):
            self.domains[i] = domain.relu(layer, nap=nap)
        for lhs in layer:
            self.reduce(lhs)
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
                        lowers = set()
                        for i, domain in enumerate(self.domains):
                            lower, upper = domain.evaluate({outA: 1, outB: -1})
                            if log:
                                diff = '{} - {}'.format(outA, outB)
                                print('{}: [{}, {}]'.format(diff, lower, upper))
                            lowers.add(lower)
                        if all({lower < 0 for lower in lowers}):
                            found = False
                if found:
                    return "{}".format(outA)
            return "?"


class SymbolicDeepPolyProductDomain(ProductDomain):
    def __init__(self, ranges):
        super().__init__(ranges, domains=[SymbolicDomain, DeepPolyDomain])
