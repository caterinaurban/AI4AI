from enum import Enum

import keras

from abstract_domains.abstract_domain import AbstractDomain
from frontend.mirror import Mirror, NAP


class Outcome(Enum):
    Infeasible = -1
    Verified = 0
    Counterexample = 1
    Unknown = 2


def bound(nn: Mirror, initial_state, initial_layer=0, nap=None):
    current: AbstractDomain = initial_state
    # current.print(label="INITIAL")
    for layer in nn.layers[initial_layer:-1]:
        if initial_layer != 0 and layer == initial_layer:
            # do not re-do the affine layer if we start somewhere deeper
            temp = current
        else:
            temp = current.affine(layer)
        current = temp.relu(layer, nap=nap)
        # current.print(label="RELU")
    final = current.affine(nn.layers[-1])
    # final.print(label="FINAL")
    activated = {lhs for lhs, flag in final.flags.items() if flag == 1}
    deactivated = {lhs for lhs, flag in final.flags.items() if flag == -1}
    found = final.outcome(nn.outputs, log=False)
    return final, activated, deactivated, found


def verify(nn, precondition, postcondition, nap: NAP = None):
    """
    :param nn: Mirror (neural network)
    :param precondition: AbstractDomain (initial analysis state)
    :param postcondition: int (wanted class index)
    :param nap: NAP (neural activation pattern)
    :return: Outcome
    """
    final, _, _, found = bound(nn, precondition, nap=nap)
    if found == '⊥':
        return Outcome.Infeasible
    if found == nn.outputs[postcondition]:
        return Outcome.Verified
    # else:
    #     # find counter-example
    #     model = keras.models.load_model('../models/wdbc/model.h5')
    #     choices = [final.bounds[ipt] for ipt in nn.inputs]
    #     import numpy as np
    #     from itertools import product
    #     all_combinations = np.array(list(product(*choices)))
    #     predictions = model.predict(all_combinations)
    #     predicted_classes = np.argmax(predictions, axis=1)
    #     all_match = np.all(predicted_classes == postcondition)
    #     print(all_match)
    return Outcome.Unknown
