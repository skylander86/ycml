import logging

import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseClassifier

__all__ = ['LabelsClassifier', 'BinaryLabelsClassifier', 'MultiLabelsClassifier']

logger = logging.getLogger(__name__)


class LabelsClassifier(BaseClassifier):
    def _fit(self, X, Y_labels, binarize_args={}, fit_args={}, **kwargs):
        Y_binarized = self.binarize_labels(Y_labels, **binarize_args)
        return self.fit_binarized(X, Y_binarized, **fit_args)
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs): raise NotImplementedError('fit_binarized is not implemented.')

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

    def binarize_labels(self, Y_labels, **kwargs): raise NotImplementedError('binarize_labels is not implemented.')

    def binarize_dicts(self, Y_dicts, default=0.0): raise NotImplementedError('binarize_dicts is not implemented.')

    def unbinarize_labels(self, Y_binarized, epsilon=0.0): raise NotImplementedError('unbinarize_labels is not implemented.')

    @property
    def classes_(self): raise NotImplementedError('classes_ is not implemented.')
#end def


class BinaryLabelsClassifier(LabelsClassifier):
    def __init__(self, pos_label=None, **kwargs):
        super(BinaryLabelsClassifier, self).__init__(**kwargs)

        self.pos_label = pos_label
    #end def

    def predict_and_proba(self, *args, **kwargs):
        Y_proba, Y_predict = super(BinaryLabelsClassifier, self).predict_and_proba(*args, **kwargs)
        binarized = kwargs.get('binarized', True)

        if binarized:
            Y_predict = Y_predict[:, 0]  # must be 0th one coz classes_ only has 1 thing

        return Y_proba, Y_predict
    #end def

    def binarize_labels(self, Y_labels, pos_label=None):
        if pos_label is None:
            pos_label = self.pos_label

        not_pos_label = 'not {}'.format(pos_label)
        Y_labels_pos = [pos_label if pos_label in Y_labels[i] else not_pos_label for i in range(Y_labels.shape[0])]
        Y_binarized = label_binarize(Y_labels_pos, classes=[pos_label, not_pos_label]).reshape(Y_labels.shape[0])

        return Y_binarized
    #end def

    def binarize_dicts(self, Y_dicts, pos_label=None, default=0.0):
        if pos_label is None:
            pos_label = self.pos_label

        return np.array([Y_dicts[i].get(pos_label, default) for i in range(Y_dicts.shape[0])])
    #end def

    def unbinarize_labels(self, Y_binarized, pos_label=None, epsilon=0.0):
        if pos_label is None:
            pos_label = self.pos_label

        return np.array([[pos_label] if Y_binarized[i] > epsilon else [] for i in range(Y_binarized.shape[0])], dtype=np.object)
    #end def

    @property
    def classes_(self):
        return [self.pos_label]
#end def


class MultiLabelsClassifier(LabelsClassifier):
    def __init__(self, n_jobs=1, ignore_labels=[], **kwargs):
        super(MultiLabelsClassifier, self).__init__(**kwargs)

        self.ignore_labels = set(ignore_labels)
    #end def

    def _fit(self, X, Y_labels, binarize_args={}, fit_args={}, **kwargs):
        if self.ignore_labels:
            Y_labels_filtered = np.empty(Y_labels.shape, dtype=np.object)
            removed_labels = 0
            for i in range(Y_labels.shape[0]):
                Y_labels_filtered[i] = [l for l in Y_labels[i] if l not in self.ignore_labels]
                removed_labels += len(Y_labels[i]) - len(Y_labels_filtered[i])
            #end for
            logger.info('{} label-instances removed from the training data.'.format(removed_labels))
        else: Y_labels_filtered = Y_labels

        self.label_binarizer_ = MultiLabelBinarizer(sparse_output=False).fit(Y_labels_filtered)
        logger.info('{} labels found in training instances.'.format(len(self.classes_)))

        return super(MultiLabelsClassifier, self)._fit(X, Y_labels, binarize_args, fit_args)
    #end def

    def binarize_labels(self, Y_labels):
        classes_ = set(self.classes_)
        missing = set(c for labels in Y_labels for c in labels) - classes_
        if missing:
            logger.debug('The following labels are ignored by the classifier: [{}]'.format(', '.join(sorted(missing))))
            logger.warning('{} labels will be ignored by the classifier.'.format(len(missing)))
        #end if

        return self.label_binarizer_.transform((filter(classes_.__contains__, labels) for labels in Y_labels))
    #end def

    def binarize_dicts(self, Y_dicts, default=0.0):
        binarized = np.fill((Y_dicts.shape[0], len(self.classes_)), default, dtype=np.float)
        classes_map = dict((c, i) for i, c in enumerate(self.clsases_))

        for i in range(Y_dicts.shape[0]):
            d = Y_dicts[i]
            for k, p in d.items():
                try: binarized[i, classes_map[k]] = p
                except IndexError: pass
            #end for
        #end for

        return binarized
    #end def

    def unbinarize_labels(self, Y_proba, epsilon=0.0, to_dict=False):
        assert len(self.classes_) == Y_proba.shape[1]
        unbinarized = np.empty(Y_proba.shape[0], dtype=np.object)
        for i in range(Y_proba.shape[0]):
            if to_dict: unbinarized[i] = dict((self.classes_[j], Y_proba[i, j]) for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon)
            else: unbinarized[i] = [self.classes_[j] for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon]
        #end for

        return unbinarized
    #end def

    @property
    def classes_(self):
        return self.label_binarizer_.classes_
    #end def
#end def
