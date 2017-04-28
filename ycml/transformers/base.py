import logging

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import Timer

__all__ = ['PureTransformer', 'identity']


logger = logging.getLogger(__name__)


# Helper class. A transformer that only does transformation and does not need to fit any internal parameters.
class PureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nparray=True, **kwargs):
        super(PureTransformer, self).__init__(**kwargs)

        self.nparray = nparray
    #end def

    def fit(self, X, y=None, **fit_params): return self

    def transform(self, X, y=None):
        timer = Timer()
        transformed = self._transform(X, y)
        if self.nparray: transformed = np.array(transformed)
        logger.debug('Done <{}> transformation{}.'.format(type(self).__name__, timer))

        return transformed
    #end def

    def _transform(self, X, y=None):
        return [self.transform_one(row) for row in X]
    #end def

    def transform_one(self, x):
        raise NotImplementedError('transform_one method needs to be implemented.')
#end class


def identity(x): return x
