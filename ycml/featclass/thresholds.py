__all__ = ['ThresholdingFeatClass']

import logging

import numpy as np

from .base import BaseFeatClass
from ..classifiers import ThresholdRescaler, BinaryLabelsClassifier
from ..classifiers import get_thresholds_from_file

logger = logging.getLogger(__name__)


class ThresholdingFeatClass(BaseFeatClass):
    def __init__(self, thresholds=None, thresholds_uri=None, **kwargs):
        super(ThresholdingFeatClass, self).__init__(**kwargs)

        if thresholds is not None and thresholds_uri is not None: raise ValueError('Only one of thresholds/thresholds_uri should be specified.')

        self.thresholds_uri = thresholds_uri

        if thresholds is None:
            if self.thresholds_uri:
                thresholds = get_thresholds_from_file(self.thresholds_uri, self.classifier.classes_)

                if len(self.classifier.classes_) == 2 and not np.isclose(thresholds.sum(), 1.0) and isinstance(self.classifier, BinaryLabelsClassifier):
                    if thresholds[0] == 0.5: thresholds[0] = 1.0 - thresholds[1]
                    if thresholds[1] == 0.5: thresholds[1] = 1.0 - thresholds[0]

                    logger.warning('Thresholds were set automatically for BinaryLabelsClassifier to {}={} and {}={}.'.format(self.classifier.classes_[0], thresholds[0], self.classifier.classes_[1], thresholds[1]))
                #end if

            else: thresholds = 0.5
        #end if

        self.rescaler = ThresholdRescaler(thresholds, len(self.classifier.classes_))
    #end def

    def predict(self, X, **kwargs):
        return self.rescaler.predict(super(ThresholdingFeatClass, self).predict_proba(X, **kwargs))
    #end def

    def predict_proba(self, X, *, rescale=True, **kwargs):
        Y_proba = super(ThresholdingFeatClass, self).predict_proba(X, **kwargs)

        if rescale:
            return self.rescaler.rescale(Y_proba)

        return Y_proba
    #end def

    def predict_and_proba(self, X, *, rescale=True, **kwargs):
        Y_proba = super(ThresholdingFeatClass, self).predict_proba(X, **kwargs)

        if rescale:
            Y_proba = self.rescaler.rescale(Y_proba)
            Y_predict = Y_proba >= 0.5
        else:
            Y_predict = self.rescaler.predict(Y_proba)
        #end if

        return Y_proba, Y_predict
    #end def

    def decision_function(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)

    def __str__(self):
        return 'ThresholdingFeatClass(featurizer={}, classifier={}, thresholds_uri={})'.format(self.featurizer, self.classifier, self.thresholds_uri if self.thresholds_uri else self.rescaler.thresholds)
    #end def
#end class
