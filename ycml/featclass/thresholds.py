import logging

from .base import BaseFeatClass
from ..classifiers import ThresholdRescaler
from ..classifiers import get_thresholds_from_file

from ..utils import uri_open, get_settings

__all__ = ['ThresholdingFeatClass']

logger = logging.getLogger(__name__)


class ThresholdingFeatClass(BaseFeatClass):
    def __init__(self, settings={}, check_environment=True, thresholds_uri=None, **kwargs):
        super(ThresholdingFeatClass, self).__init__(settings=settings, check_environment=check_environment, **kwargs)

        sources = ('env', settings) if check_environment else (settings,)

        self.thresholds_uri = get_settings(key='thresholds_uri', sources=sources, raise_on_missing=False) if thresholds_uri is None else thresholds_uri

        if self.thresholds_uri:
            with uri_open(self.thresholds_uri) as f:
                thresholds = get_thresholds_from_file(f, self.classifier_.classes_)
        else: thresholds = 0.5

        self.rescaler = ThresholdRescaler(thresholds, len(self.classifier_.classes_))
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
        return 'ThresholdingFeatClass(featurizer_uri={}, classifier_uri={}, thresholds_uri={})'.format(self.featurizer_uri, self.classifier_uri, self.thresholds_uri)
    #end def
#end class
