import logging

import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseClassifier

__all__ = ['LabelsClassifier', 'BinaryLabelsClassifier', 'MultiLabelsClassifier', 'MulticlassLabelsClassifier']

logger = logging.getLogger(__name__)


class LabelsClassifier(BaseClassifier):
    def __init__(self, include=[], exclude=[], **kwargs):
        super(LabelsClassifier, self).__init__(**kwargs)

        self.exclude = set(exclude)
        self.include = set(include)
    #end def

    def _fit(self, X, Y_labels, *, binarize_args={}, fit_args={}, **kwargs):
        Y_binarized = self.binarize_labels(Y_labels, **binarize_args)
        return self.fit_binarized(X, Y_binarized, **fit_args)
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs): raise NotImplementedError('fit_binarized is not implemented.')

    def predict(self, X_featurized, binarized=True, **kwargs):
        Y_predict_binarized = super(LabelsClassifier, self).predict(X_featurized, **kwargs)
        if binarized: return Y_predict_binarized

        return self.unbinarize_labels(Y_predict_binarized)
    #end def

    def predict_and_proba(self, X_featurized, *, binarized=True, **kwargs):
        Y_proba, Y_predict_binarized = super(LabelsClassifier, self).predict_and_proba(X_featurized, **kwargs)
        if binarized: return Y_proba, Y_predict_binarized

        return Y_proba, self.unbinarize_labels(Y_predict_binarized)
    #end def

    def binarize_labels(self, Y_labels, **kwargs): raise NotImplementedError('binarize_labels is not implemented.')

    def binarize_dicts(self, Y_dicts, *, default=0.0): raise NotImplementedError('binarize_dicts is not implemented.')

    def unbinarize_labels(self, Y_binarized, *, epsilon=0.0): raise NotImplementedError('unbinarize_labels is not implemented.')

    def _filter_labels(self, Y_labels):
        if self.exclude or self.include:
            Y_labels_filtered = np.empty(Y_labels.shape, dtype=np.object)
            removed_labels = 0
            for i in range(Y_labels.shape[0]):
                Y_labels_filtered[i] = [l for l in Y_labels[i] if (l in self.include or not self.include) and l not in self.exclude]
                removed_labels += len(Y_labels[i]) - len(Y_labels_filtered[i])
            #end for
            logger.info('{} label-instances removed from the training data.'.format(removed_labels))
        else: Y_labels_filtered = Y_labels

        return Y_labels_filtered
    #end def
#end def


class BinaryLabelsClassifier(LabelsClassifier):
    def __init__(self, pos_label, *, not_pos_label=None, **kwargs):
        super(BinaryLabelsClassifier, self).__init__(**kwargs)

        self.pos_label = pos_label
        self.not_pos_label = 'not ' + pos_label if not_pos_label is None else not_pos_label
    #end def

    def predict_and_proba(self, X_featurized, *, binarized=True, **kwargs):
        Y_proba, Y_predict = super(BinaryLabelsClassifier, self).predict_and_proba(X_featurized, binarized=binarized, **kwargs)
        if Y_proba.ndim == 1:
            Y_proba_2d = np.zeros((Y_proba.shape[0], 2), dtype=np.float64)
            Y_proba_2d[:, 0] = Y_proba
            Y_proba_2d[:, 1] = 1.0 - Y_proba
            Y_proba = Y_proba_2d
        #end if

        if binarized:
            Y_predict = Y_predict[:, 0]  # must be 0th one coz classes_ only has 1 thing

        return Y_proba, Y_predict
    #end def

    def binarize_labels(self, Y_labels):
        if Y_labels.shape[0] == 0:
            return np.zeros((0, 1))

        Y_labels_pos = [self.pos_label if self.pos_label in Y_labels[i] else self.not_pos_label for i in range(Y_labels.shape[0])]
        Y_binarized = label_binarize(Y_labels_pos, classes=[self.not_pos_label, self.pos_label]).reshape(Y_labels.shape[0])  # 1 is for pos label, 0 otherwise

        return Y_binarized
    #end def

    def binarize_dicts(self, Y_dicts, *, default=0.0):
        return np.array([Y_dicts[i].get(self.pos_label, default) for i in range(Y_dicts.shape[0])])
    #end def

    def unbinarize_labels(self, Y_proba, *, epsilon=0.0, to_dict=False):
        unbinarized = np.empty(Y_proba.shape[0], dtype=np.object)
        if len(Y_proba.shape) == 1:
            for i in range(Y_proba.shape[0]):
                if to_dict: unbinarized[i] = dict([(self.classes_[Y_proba[i]], float(Y_proba[i]))])
                else: unbinarized[i] = [self.classes_[Y_proba[i]]]
            #end for
        else:
            Y_argmax = np.argmax(Y_proba, axis=1)
            for i in range(Y_proba.shape[0]):
                if to_dict: unbinarized[i] = dict((self.classes_[j], float(Y_proba[i, j])) for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon)
                else: unbinarized[i] = [self.classes_[Y_argmax[i]]]
            #end for
        #end if

        return unbinarized
    #end def

    @property
    def classes_(self):
        return [self.pos_label, self.not_pos_label]
#end def


# TODO: use multilabel format and inherit multilabelclassifier
class MulticlassLabelsClassifier(LabelsClassifier):
    def __init__(self, **kwargs):
        super(MulticlassLabelsClassifier, self).__init__(**kwargs)
    #end def

    def _fit(self, X, Y_labels, *, binarize_args={}, fit_args={}, **kwargs):
        Y_labels_filtered = self._filter_labels(Y_labels)
        Y_labels_filtered = np.array([Y_labels_filtered[i][0] if Y_labels_filtered[i] else '<none>' for i in range(Y_labels_filtered.shape[0])])

        self.label_encoder_ = LabelEncoder().fit(Y_labels_filtered)
        logger.info('{} labels found in training instances.'.format(len(self.classes_)))

        if not len(self.classes_): raise ValueError('There are no labels available for fitting model.')

        return super(MulticlassLabelsClassifier, self)._fit(X, Y_labels, binarize_args=binarize_args, fit_args=fit_args, **kwargs)
    #end def

    def predict_and_proba(self, X_featurized, *, binarized=True, **kwargs):
        Y_proba = self.predict_proba(X_featurized, **kwargs)
        Y_predict_binarized = np.argmax(Y_proba, axis=1)

        if binarized: return Y_proba, Y_predict_binarized

        return Y_proba, self.unbinarize_labels(Y_predict_binarized)
    #end def

    def predict_proba(self, X_featurized, **kwargs):
        if 'thresholds' in kwargs: logger.warning('thresholds argument is ignored by <{}>.'.format(self))
        if 'denominators' in kwargs: logger.warning('denominators argument is ignored by <{}>.'.format(self))
        if 'rescale' in kwargs:
            del kwargs['rescale']
            logger.warning('rescale argument is ignored by <{}>.'.format(self))
        #end if

        return super(MulticlassLabelsClassifier, self).predict_proba(X_featurized, rescale=False, **kwargs)
    #end def

    def binarize_labels(self, Y_labels):
        classes_ = set(self.classes_)

        Y_labels_filtered = np.empty(Y_labels.shape[0], dtype=np.object)
        for i in range(Y_labels.shape[0]):
            labels = list(filter(classes_.__contains__, Y_labels[i]))
            Y_labels_filtered[i] = labels[0] if labels else '<none>'

        return self.label_encoder_.transform(Y_labels_filtered)
    #end def

    def binarize_dicts(self, Y_dicts, *, default=0.0):
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

    def unbinarize_labels(self, Y_proba, *, epsilon=0.0, to_dict=False):
        unbinarized = np.empty(Y_proba.shape[0], dtype=np.object)
        if len(Y_proba.shape) == 1:
            for i in range(Y_proba.shape[0]):
                if to_dict: unbinarized[i] = dict([(self.classes_[Y_proba[i]], float(Y_proba[i]))])
                else: unbinarized[i] = [self.classes_[Y_proba[i]]]
            #end for
        else:
            Y_argmax = np.argmax(Y_proba, axis=1)
            for i in range(Y_proba.shape[0]):
                if to_dict: unbinarized[i] = dict((self.classes_[j], float(Y_proba[i, j])) for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon)
                else: unbinarized[i] = [self.classes_[Y_argmax[i]]]
            #end for
        #end if

        return unbinarized
    #end def

    @property
    def classes_(self):
        return self.label_encoder_.classes_
#end def


class MultiLabelsClassifier(LabelsClassifier):
    def __init__(self, n_jobs=1, include=[], exclude=[], **kwargs):
        super(MultiLabelsClassifier, self).__init__(**kwargs)

        self.exclude = set(exclude)
        self.include = set(include)
    #end def

    def _fit(self, X, Y_labels, *, binarize_args={}, fit_args={}, **kwargs):
        Y_labels_filtered = self._filter_labels(Y_labels)
        self.label_binarizer_ = MultiLabelBinarizer(sparse_output=False).fit(Y_labels_filtered)
        logger.info('{} labels found in training instances.'.format(len(self.classes_)))

        if not len(self.classes_): raise ValueError('There are no labels available for fitting model.')

        return super(MultiLabelsClassifier, self)._fit(X, Y_labels, binarize_args=binarize_args, fit_args=fit_args, **kwargs)
    #end def

    def binarize_labels(self, Y_labels):
        classes_ = set(self.classes_)

        return self.label_binarizer_.transform((filter(classes_.__contains__, labels) for labels in Y_labels))
    #end def

    def binarize_dicts(self, Y_dicts, *, default=0.0):
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

    def unbinarize_labels(self, Y_proba, *, epsilon=0.0, to_dict=False):
        assert len(self.classes_) == Y_proba.shape[1]
        unbinarized = np.empty(Y_proba.shape[0], dtype=np.object)
        for i in range(Y_proba.shape[0]):
            if to_dict: unbinarized[i] = dict((self.classes_[j], float(Y_proba[i, j])) for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon)
            else: unbinarized[i] = [self.classes_[j] for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon]
        #end for

        return unbinarized
    #end def

    @property
    def classes_(self):
        return self.label_binarizer_.classes_
    #end def
#end def
