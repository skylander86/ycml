from io import BytesIO
from datetime import datetime
import logging
import pickle
import tarfile
import time
from uuid import uuid4

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from ..utils import load_dictionary_from_file
from ..utils import Timer
from ..utils import parse_n_jobs

__all__ = ['BaseClassifier', 'load_classifier', 'get_thresholds_from_file']

logger = logging.getLogger(__name__)


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=1, **kwargs):
        super(BaseClassifier, self).__init__()

        self.n_jobs = parse_n_jobs(n_jobs)
        logger.info('Using n_jobs={} for <{}>.'.format(self.n_jobs, self.name))
    #end def

    def fit(self, X, Y, **kwargs):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(self.name, self.uuid_))

        self.fitted_at_ = datetime.utcnow()

        timer = Timer()
        self._fit(X, Y, **kwargs)
        logger.info('{} fitting on {} instances complete {}.'.format(self.name, X.shape[0], timer))

        return self
    #end def

    def _fit(self, *args, **kwargs): raise NotImplementedError('_fit is not implemented.')

    def predict_proba(self, X_featurized, *, thresholds=0.5, denominators=None, rescale=False, **kwargs):
        timer = Timer()
        Y_proba = self._predict_proba(X_featurized, **kwargs)
        logger.debug('Computed prediction probabilities on {} instances {}.'.format(X_featurized.shape[0], timer))

        if not rescale: return Y_proba

        if isinstance(thresholds, (float, np.float)):
            thresholds = np.full(Y_proba.shape[1], thresholds)

        if denominators is None:
            denominators = 2.0 * (1.0 - thresholds)

        rescaled = np.zeros(Y_proba.shape)
        for i in range(Y_proba.shape[0]):
            for k in range(Y_proba.shape[1]):
                if Y_proba[i, k] > thresholds[k]: rescaled[i, k] = 0.5 + ((Y_proba[i, k] - thresholds[k]) / denominators[k])
                else: rescaled[i, k] = Y_proba[i, k] / (thresholds[k] + thresholds[k])
            #end for
        #end for

        return rescaled
    #end def

    def _predict_proba(self, X_featurized, **kwargs):
        raise NotImplementedError('_predict_proba is not implemented.')

    def predict(self, X_featurized, thresholds=0.5, **kwargs):
        _, Y_predict = self.predict_and_proba(X_featurized, thresholds=thresholds, rescale=False, **kwargs)

        return Y_predict
    #end def

    def predict_and_proba(self, X_featurized, *, thresholds=0.5, rescale=False, **kwargs):
        Y_proba = self.predict_proba(X_featurized, thresholds=thresholds, rescale=rescale, **kwargs)

        if rescale: Y_predict = Y_proba >= 0.5
        else:
            Y_predict = np.zeros(Y_proba.shape, dtype=np.bool)
            if thresholds is None: thresholds = np.full(Y_proba.shape[1], 0.5)
            for i in range(Y_proba.shape[0]):
                Y_predict[i, :] = Y_proba[i, :] >= thresholds
        #end def

        return Y_proba, Y_predict
    #end def

    def decision_function(self, *args, **kwargs): return self.predict_proba(*args, **kwargs)

    def save(self, f):
        if not hasattr(self, 'uuid_'):
            raise NotFittedError('This featurizer is not fitted yet.')

        with tarfile.open(fileobj=f, mode='w') as tf:
            with BytesIO() as model_f:
                pickle.dump(self, model_f, protocol=4)
                model_data = model_f.getvalue()
                model_f.seek(0)
                model_tarinfo = tarfile.TarInfo(name='model.pkl')
                model_tarinfo.size = len(model_data)
                model_tarinfo.mtime = int(time.time())
                tf.addfile(tarinfo=model_tarinfo, fileobj=model_f)
            #end with

            self.save_to_tarfile(tf)
        #end with

        logger.info('{} saved to <{}>.'.format(self, f.name))

        return self
    #end def

    def save_to_tarfile(self, tf): return self

    def load_from_tarfile(self, tf): return self

    @property
    def uuid(self):
        return self.uuid_
    #end def

    @property
    def name(self):
        return type(self).__name__

    @property
    def classes_(self): raise NotImplementedError('classes_ is not implemented.')

    def __str__(self):
        return '{}(UUID={})'.format(self.name, self.uuid_ if hasattr(self, 'uuid_') else 'None')
#end class


def load_classifier(f):
    with tarfile.open(fileobj=f, mode='r') as tf:
        classifier = pickle.load(tf.extractfile('model.pkl'))
        classifier.load_from_tarfile(tf)
    #end with

    logger.info('{} loaded from <{}>.'.format(classifier, f.name))

    return classifier
#end def


def get_thresholds_from_file(f, classes, *, default=0.5):
    d = load_dictionary_from_file(f)
    return np.array([float(d.get(c, default)) for c in classes])
#end def
