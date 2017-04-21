from datetime import datetime
import logging
import pickle
from uuid import uuid4

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import label_binarize

from ..utils import load_dictionary_from_file
from ..utils import Timer

__all__ = ['BaseClassifier', 'BinaryClassifier', 'load_classifier', 'get_thresholds_from_file']

logger = logging.getLogger(__name__)


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=1, **kwargs):
        super(BaseClassifier, self).__init__(**kwargs)

        self.n_jobs = n_jobs
    #end def

    def fit(self, X, Y, **kwargs):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(type(self).__name__, self.uuid_))

        self.fitted_at_ = datetime.utcnow()

        timer = Timer()
        self._fit(X, Y, **kwargs)
        logger.info('{} fitting on {} instances complete {}.'.format(type(self).__name__, X.shape[0], timer))

        return self
    #end def

    def _fit(self, *args, **kwargs):
        raise NotImplementedError('_fit is not implemented.')

    def predict_proba(self, X_featurized, thresholds=0.5, denominators=None, rescale=False, **kwargs):
        timer = Timer()
        Y_proba = self._predict_proba(X_featurized, **kwargs)
        logger.debug('Computed prediction probabilities on {} instances {}.'.format(X_featurized.shape[0], timer))

        if not rescale: return Y_proba

        n_classes = len(self.classes_)
        if isinstance(thresholds, (float, np.float)):
            thresholds = np.full(n_classes, thresholds)

        if denominators is None:
            denominators = 2.0 * (1.0 - thresholds)

        rescaled = np.zeros(Y_proba.shape)
        for i in range(Y_proba.shape[0]):
            for k in range(n_classes):
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

    def predict_and_proba(self, X_featurized, thresholds=0.5, denominators=None, rescale=False, **kwargs):
        Y_proba = self.predict_proba(X_featurized, thresholds=thresholds, denominators=denominators, rescale=rescale, **kwargs)

        if rescale: Y_predict = Y_proba >= 0.5
        else:
            Y_predict = np.zeros(Y_proba.shape, dtype=np.bool)
            for i in range(Y_proba.shape[0]):
                Y_predict[i, :] = Y_proba[i, :] >= thresholds
        #end def

        return Y_proba, Y_predict
    #end def

    def decision_function(self, *args, **kwargs): return self.predict_proba(*args, **kwargs)

    def save(self, f):
        if not hasattr(self, 'uuid_'):
            raise NotFittedError('This featurizer is not fitted yet.')

        pickle.dump(self, f, protocol=4)
        logger.info('Saved {} to <{}>.'.format(self, f.name))
    #end def

    @property
    def uuid(self):
        return self.uuid_
    #end def

    def __str__(self):
        return '{}(UUID={})'.format(type(self).__name__, self.uuid_ if hasattr(self, 'uuid_') else 'None')
#end class


class LabelsClassifier(BaseClassifier):
    def fit_binarized(self, X_featurized, Y_labels, **kwargs): raise NotImplementedError('fit_binarized is not implemented.')

    def predict(self, X_featurized, binarized=True, **kwargs):
        Y_predict_binarized = super(LabelsClassifier, self).predict(X_featurized, **kwargs)
        if binarized: return Y_predict_binarized

        return self.unbinarize_labels(Y_predict_binarized)
    #end def

    def predict_and_proba(self, X_featurized, binarized=True, **kwargs):
        Y_proba, Y_predict_binarized = super(LabelsClassifier, self).predict_and_proba(X_featurized, **kwargs)
        if binarized: return Y_proba, Y_predict_binarized

        return Y_proba, self.unbinarize_labels(Y_predict_binarized)
    #end def

    def binarize_labels(self, Y_labels): raise NotImplementedError('binarize_labels is not implemented.')

    def unbinarize_labels(self, Y_binarized, epsilon=0.0): raise NotImplementedError('unbinarize_labels is not implemented.')

    @property
    def classes_(self): raise NotImplementedError('classes_ is not implemented.')
#end def


class BinaryClassifier(LabelsClassifier):
    def _fit(self, X_featurized, Y_labels, pos_labels=[], **kwargs):
        if isinstance(pos_labels, (list, tuple)):
            if len(pos_labels) != 1:
                raise ValueError('pos_labels for BinaryClassifier must be of length = 1.')
            self.pos_label_ = pos_labels[0]
        else:
            self.pos_label_ = pos_labels
        #end if

        not_pos_label = 'not {}'.format(self.pos_label_)
        Y_labels_pos = [self.pos_label_ if self.pos_label_ in Y_labels[i] else not_pos_label for i in range(Y_labels.shape[0])]
        Y_binarized = label_binarize(Y_labels_pos, classes=[self.pos_label_, not_pos_label]).reshape(Y_labels.shape[0])

        self.fit_binarized(X_featurized, Y_binarized, **kwargs)

        return self
    #end def

    def predict(self, X_featurized, binarized=True, **kwargs):
        Y_predict_binarized = super(BinaryClassifier, self).predict(X_featurized, **kwargs)
        if binarized: return Y_predict_binarized

        return self.unbinarize_labels(Y_predict_binarized)
    #end def

    def predict_and_proba(self, *args, **kwargs):
        binarized = kwargs.pop('binarized', True)
        Y_proba, Y_predict = super(BinaryClassifier, self).predict_and_proba(*args, binarized=binarized, **kwargs)
        if binarized:
            Y_predict = Y_predict[:, 0]

        return Y_proba, Y_predict
    #end def

    def binarize_labels(self, Y_labels):
        not_pos_label = 'not {}'.format(self.pos_label_)
        Y_labels_pos = [self.pos_label_ if self.pos_label_ in Y_labels[i] else not_pos_label for i in range(Y_labels.shape[0])]
        return label_binarize(Y_labels_pos, classes=[self.pos_label_, not_pos_label])
    #end def

    def unbinarize_labels(self, Y_binarized, epsilon=0.0):
        return np.array([[self.pos_label_] if Y_binarized[i] > 0 else [] for i in range(Y_binarized.shape[0])], dtype=np.object)

    @property
    def classes_(self): return [self.pos_label_]
#end def


def load_classifier(f):
    classifier = pickle.load(f)
    logger.info('{} loaded from <{}>.'.format(classifier, f.name))

    return classifier
#end def


def get_thresholds_from_file(f, classes, default=0.5):
    d = load_dictionary_from_file(f)
    return np.array([float(d.get(c, default)) for c in classes])
#end def
