__all__ = ['PureTransformer', 'identity']

import logging

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import Timer

logger = logging.getLogger(__name__)


# Helper class. A transformer that only does transformation and does not need to fit any internal parameters.
class PureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nparray=True, nparray_dtype=None, generator=False, **kwargs):
        super(PureTransformer, self).__init__()

        self.nparray = nparray
        self.nparray_dtype = nparray_dtype
        self.generator = generator

        if self.nparray and self.generator:
            raise ValueError('nparray and generator option cannot both be True.')
    #end def

    def fit(self, *args, **fit_params):
        return self

    def transform(self, X, y=None, **kwargs):
        timer = Timer()
        transformed = self._transform(X, y=y, **kwargs)

        if self.nparray:
            nparray_dtype = getattr(self, 'nparray_dtype', None)
            if nparray_dtype:
                transformed = np.array(transformed, dtype=nparray_dtype)
            else:
                transformed = np.array(transformed)
                if transformed.ndim == 1:
                    transformed = transformed.reshape(transformed.shape[0], 1)
            #end if
        #end if

        logger.debug('Done <{}> transformation {}.'.format(type(self).__name__, timer))

        return transformed
    #end def

    def _transform(self, X, y=None, **kwargs):
        if getattr(self, 'generator', False):
            return (self.transform_one(row, **kwargs) for row in X)

        return [self.transform_one(row, **kwargs) for row in X]
    #end def

    def transform_one(self, x, **kwargs):
        raise NotImplementedError('transform_one method needs to be implemented.')
#end class


def identity(x): return x
