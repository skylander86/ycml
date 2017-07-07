from collections import Counter, defaultdict
from datetime import datetime
import logging
import pickle as pickle
from uuid import uuid4

import numpy as np
from numpy.lib.npyio import NpzFile

from scipy.sparse import csr_matrix, issparse

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from tabulate import tabulate

from ..utils import Timer

__all__ = ['BaseFeaturizer', 'load_featurized', 'save_featurized', 'load_featurizer']

logger = logging.getLogger(__name__)


class BaseFeaturizer(Pipeline):
    def __init__(self, n_jobs=1, **kwargs):
        super(BaseFeaturizer, self).__init__(**kwargs)
        self.n_jobs = n_jobs
    #end def

    def fit(self, *args, **kwargs):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(type(self).__name__, self.uuid_))

        timer = Timer()
        super(BaseFeaturizer, self).fit(*args, **kwargs)
        logger.info('Done fitting {} {}.'.format(type(self).__name__, timer))

        self._post_fit(*args, **kwargs)

        return self
    #end def

    def fit_transform(self, X):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(type(self).__name__, self.uuid_))

        timer = Timer()
        X_featurized = super(BaseFeaturizer, self).fit_transform(X)
        logger.info('Done fitting {} {}.'.format(type(self).__name__, timer))

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

    def _post_fit(self, *args, **kwargs): return self
#end def


def load_featurizer(f):
    featurizer = pickle.load(f)
    logger.info('Loaded {} from <{}>.'.format(featurizer, f.name))
    if not isinstance(featurizer, BaseFeaturizer):
        logger.warning('{} is not an instance of BaseFeaturizer. Perhaps you are loading the wrong file?'.format(featurizer))

    return featurizer
#end def


def save_featurized(f, X_featurized, *, Y_labels=None, featurized_at=datetime.utcnow(), **kwargs):
    if issparse(X_featurized):
        np.savez_compressed(f, X_featurized_data=X_featurized.data, X_featurized_indices=X_featurized.indices, X_featurized_indptr=X_featurized.indptr, X_featurized_shape=X_featurized.shape, Y_labels=Y_labels, featurized_at=featurized_at, **kwargs)  # Always use compression
    else:
        np.savez_compressed(f, X_featurized=X_featurized, Y_labels=Y_labels, featurized_at=featurized_at, **kwargs)  # Always use compression
    #end if

    logger.info('Saved {} featurized instances and its metadata to <{}>.'.format(X_featurized.shape[0], f.name))
#end def


def load_featurized(f, keys=[], raise_on_missing=True):
    timer = Timer()

    if f is None:
        logger.warning('No featurized file is specified. Will return empty instances.')
        o = defaultdict(None)
        o['X_featurized'] = np.empty((0, 1))
        o['Y_labels'] = np.empty((0, 1))
        f = type('test', (), {'name': 'file not specified'})()  # Hack to get around displaying empty file name
    else:
        o = np.load(f)
        if not isinstance(o, NpzFile):
            logger.warning('<{}> is not an instance of NpzFile. Perhaps you are loading the wrong file?'.format(f.name))
    #end if

    if not keys or 'X_featurized' in keys:
        if 'X_featurized_data' in o: X_featurized = csr_matrix((o['X_featurized_data'], o['X_featurized_indices'], o['X_featurized_indptr']), shape=o['X_featurized_shape'])
        else: X_featurized = o['X_featurized']

        logger.info('Loaded {} featurized instances from <{}> {}.'.format(X_featurized.shape[0], f.name, timer))
    #end if

    if keys and raise_on_missing:
        for k in keys:
            if k not in o and k not in ['X_featurized']:
                raise ValueError('{} not found in <{}>.'.format(k, f.name))
    #end if

    Y_labels = None
    if keys:
        d = tuple([X_featurized if k == 'X_featurized' else o[k] for k in keys])
        if 'Y_labels' in keys: Y_labels = d[keys.index('Y_labels')]
    else:
        d = dict((k, o[k]) for k in o.keys() if k not in {'X_featurized_data', 'X_featurized_indices', 'X_featurized_shape', 'X_featurized_indptr'})
        Y_labels = d.get('Y_labels')
    #end if

    if Y_labels is not None:
        freq = Counter(label for labels in Y_labels for label in labels)
        freq['<none>'] = sum(1 for labels in Y_labels if not labels)
        if freq['<none>'] == 0: del freq['<none>']

        logger.info('Label frequencies for <{}>:\n{}'.format(f.name, tabulate(freq.most_common() + [('Labels total', sum(freq.values())), ('Cases total', len(Y_labels))], headers=('Label', 'Freq'), tablefmt='psql')))
    #end for

    if isinstance(d, tuple): return d
    return X_featurized, d
#end def
