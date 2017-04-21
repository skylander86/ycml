from datetime import datetime
import logging
import pickle as pickle
import time
from uuid import uuid4

import numpy as np

from scipy.sparse import csr_matrix, issparse

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

__all__ = ['BaseFeaturizer', 'load_featurized', 'save_featurized', 'load_featurizer']

logger = logging.getLogger(__name__)


class BaseFeaturizer(Pipeline):
    def __init__(self, **kwargs):
        super(BaseFeaturizer, self).__init__(**kwargs)
    #end def

    def fit(self, X):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(type(self).__name__, self.uuid_))

        start_time = time.time()
        super(BaseFeaturizer, self).fit(X)
        logger.info('Done fitting {} (took {:.3f} seconds).'.format(type(self).__name__, time.time() - start_time))

        self._post_fit()

        return self
    #end def

    def fit_transform(self, X):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(type(self).__name__, self.uuid_))

        start_time = time.time()
        X_featurized = super(BaseFeaturizer, self).fit_transform(X)
        logger.info('Done fitting {} (took {:.3f} seconds).'.format(type(self).__name__, time.time() - start_time))

        self._post_fit()

        return X_featurized
    #end def

    def save(self, f):
        if not hasattr(self, 'uuid_'):
            raise NotFittedError('This featurizer is not fitted yet.')

        pickle.dump(self, f, protocol=4)
        logger.info('Saved {} to <{}>.'.format(self, f.name))
    #end def

    def __str__(self):
        return '{}(uuid={})'.format(type(self).__name__, self.uuid_ if hasattr(self, 'uuid_') else 'None')
    #end def

    @property
    def uuid(self):
        if not hasattr(self, 'uuid_'):
            raise NotFittedError('This featurizer is not fitted yet.')

        return self.uuid_
    #end def

    def _post_fit(self): return
#end def


def load_featurizer(f):
    featurizer = pickle.load(f)
    logger.info('Loaded {} from <{}>.'.format(featurizer, f.name))

    return featurizer
#end def


def save_featurized(f, X_featurized, Y_labels=None, **kwargs):
    assert issparse(X_featurized)

    np.savez_compressed(f, X_featurized_data=X_featurized.data, X_featurized_indices=X_featurized.indices, X_featurized_indptr=X_featurized.indptr, X_featurized_shape=X_featurized.shape, Y_labels=Y_labels, featurized_at=datetime.utcnow(), **kwargs)  # Always use compression
    logger.info('Saved {} featurized instances and its metadata to <{}>.'.format(X_featurized.shape[0], f.name))
#end def


def load_featurized(f, keys=[], raise_on_missing=True):
    o = np.load(f)

    if not keys or 'X_featurized' in keys:
        X_featurized = csr_matrix((o['X_featurized_data'], o['X_featurized_indices'], o['X_featurized_indptr']), shape=o['X_featurized_shape'])
        logger.info('Loaded {} featurized instances from <{}>.'.format(X_featurized.shape[0], f.name))
    #end if

    if keys and raise_on_missing:
        for k in keys:
            if k not in o:
                raise ValueError('{} not found in <{}>.'.format(k, f.name))
    #end if

    if keys: return [X_featurized if k == 'X_featurized' else o.get(k) for k in keys]

    d = dict((k, o[k]) for k in o.keys() if k not in {'X_featurized_data', 'X_featurized_indices', 'X_featurized_shape', 'X_featurized_indptr'})
    return X_featurized, d
#end def
