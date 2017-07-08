__all__ = ['ThresholdRescaler', 'get_thresholds_from_file']

import numpy as np

from ..utils import load_dictionary_from_file


class ThresholdRescaler(object):
    def __init__(self, thresholds, n_classes=None):
        if isinstance(thresholds, float): self.thresholds = np.full(n_classes, thresholds)
        elif isinstance(thresholds, (list, tuple)): self.thresholds = np.array(thresholds)
        else: self.thresholds = thresholds

        if self.thresholds.ndim == 2:
            self.thresholds = self.thresholds[:, 0]

        if self.thresholds.ndim != 1: raise ValueError('`thresholds` should be a 1D array.')

        if n_classes is None: n_classes = self.thresholds.shape[0]
        else: assert n_classes == self.thresholds.shape[0]

        self.denominators = 2.0 * (1.0 - self.thresholds)
    #end def

    def rescale(self, Y_proba):
        return rescale_proba_with_thresholds(Y_proba, self.thresholds, denominators=self.denominators)
    #end def

    def predict(self, Y_proba):
        Y_predict = np.zeros(Y_proba.shape, dtype=np.bool)
        for i in range(Y_proba.shape[0]):
            Y_predict[i, :] = Y_proba[i, :] > self.thresholds

        return Y_predict
    #end def
#end class


def rescale_proba_with_thresholds(Y_proba, thresholds, *, denominators=None):
    assert Y_proba.shape[1] == thresholds.shape[0]

    if denominators is None: denominators = 2.0 * (1.0 - thresholds)

    rescaled = np.zeros(Y_proba.shape)
    for i in range(Y_proba.shape[0]):
        for k in range(Y_proba.shape[1]):
            if Y_proba[i, k] >= thresholds[k]: rescaled[i, k] = 0.5 + ((Y_proba[i, k] - thresholds[k]) / denominators[k])
            else: rescaled[i, k] = Y_proba[i, k] / (thresholds[k] * 2.0)
        #end for
    #end for

    return rescaled
#end def


def get_thresholds_from_file(f, classes, *, default=0.5):
    d = load_dictionary_from_file(f)
    return np.array([float(d.get(c, default)) for c in classes])
#end def
