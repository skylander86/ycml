import logging
import time

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['PureTransformer']


logger = logging.getLogger(__name__)


# Helper class. A transformer that only does transformation and does not need to fit any internal parameters.
class PureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(PureTransformer, self).__init__()
    #end def

    def fit(self, X, y=None, **fit_params): return self

    def transform(self, X, y=None):
        start_time = time.time()
        transformed = self._transform(X, y)
        logger.debug('Done <{}> transformation (took {:.3f} seconds).'.format(type(self).__name__, time.time() - start_time))

        return transformed
    #end def

    def _transform(self, X, y=None):
        return [self.transform_one(row) for row in X]

    def transform_one(self, x):
        raise NotImplementedError('transform_one method needs to be implemented.')
#end class
