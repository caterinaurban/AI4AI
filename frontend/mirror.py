from enum import Enum
from typing import List, Dict


class Activations(Enum):
    RELU = 0
    SIGMOID = 1


class Mirror:
    """ Internal Neural Network Representation """

    def __init__(self, inputs, activations, layers, outputs):
        self._inputs: List[str] = inputs
        self._activations: Dict[str, Activations] = activations
        self._layers: List[Dict[str, Dict[str, float]]] = layers
        self._outputs: List[str] = outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def activations(self):
        return self._activations

    @property
    def layers(self):
        return self._layers

    @property
    def outputs(self):
        return self._outputs


class Status(Enum):
    ACTIVE = 1
    INACTIVE = -1
    UNKNOWN = 0


class NAP:
    """ Neural Activation Pattern """

    def __init__(self, activations: List[str]):
        self._pattern: Dict[str, Status] = dict()
        for relu in activations:
            self._pattern[relu] = Status.UNKNOWN
        self._active_count = 0
        self._inactive_count = 0

    def __contains__(self, relu):
        return self.is_active(relu) or self.is_inactive(relu)

    @property
    def pattern(self):
        return self._pattern

    @property
    def size(self):
        return self._active_count + self._inactive_count

    @property
    def active(self) -> List[str]:
        active = list()
        for relu, status in self.pattern.items():
            if self.is_active(relu):
                active.append(relu)
        return active

    @property
    def inactive(self) -> List[str]:
        inactive = list()
        for relu, status in self.pattern.items():
            if self.is_inactive(relu):
                inactive.append(relu)
        return inactive

    @property
    def unknown(self) -> List[str]:
        unknown = list()
        for relu, status in self.pattern.items():
            if self.is_unknown(relu):
                unknown.append(relu)
        return unknown

    def is_active(self, relu):
        return self.pattern[relu] == Status.ACTIVE

    def is_inactive(self, relu):
        return self.pattern[relu] == Status.INACTIVE

    def is_unknown(self, relu):
        return self.pattern[relu] == Status.UNKNOWN

    def make_active(self, relu):
        if self.is_inactive(relu):
            self._inactive_count = self._inactive_count - 1
        self.pattern[relu] = Status.ACTIVE
        self._active_count = self._active_count + 1

    def make_inactive(self, relu):
        if self.is_active(relu):
            self._active_count = self._active_count - 1
        self.pattern[relu] = Status.INACTIVE
        self._inactive_count = self._inactive_count + 1

    def make_unknown(self, relu):
        if self.is_active(relu):
            self._active_count = self._active_count - 1
        if self.is_inactive(relu):
            self._inactive_count = self._inactive_count - 1
        self.pattern[relu] = Status.UNKNOWN

    def __repr__(self):
        size = len(self.pattern)
        active = self.active
        a = 'Active ({} of {}): {}'.format(len(active), size, active)
        # a = 'Active ({} of {})'.format(len(active), size)
        inactive = self.inactive
        i = 'Inactive ({} of {}): {}'.format(len(inactive), size, inactive)
        # i = 'Inactive ({} of {})'.format(len(inactive), size)
        unknown = self.unknown
        # u = 'Unknown ({} of {}): {}'.format(len(unknown), size, unknown)
        u = 'Unknown ({} of {})'.format(len(unknown), size)
        return '{}\n{}\n{}'.format(a, i, u)
