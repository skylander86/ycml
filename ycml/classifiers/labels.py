import logging

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseClassifier

__all__ = ['BinaryLabelsClassifier', 'MultiLabelsClassifier', 'MulticlassLabelsClassifier']

logger = logging.getLogger(__name__)


class LabelsClassifierMixin(BaseClassifier):
    def __init__(self, include=[], exclude=[], **kwargs):
        super(LabelsClassifierMixin, self).__init__(**kwargs)

        self.exclude = set(exclude)
        self.include = set(include)
    #end def

    def _fit(self, X, Y_labels, *, validation_data=None, binarize_args={}, fit_args={}, **kwargs):
        Y_binarized = self.binarize_labels(Y_labels, **binarize_args, **kwargs)
        if validation_data is not None:
            X_validation, Y_validation = validation_data
            validation_data = (X_validation, self.binarize_labels(Y_validation, **binarize_args, **kwargs))
            logger.debug('Binarized validation labels.')
        #end if

        return self.fit_binarized(X, Y_binarized, validation_data=validation_data, **fit_args, **kwargs)
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs): raise NotImplementedError('fit_binarized is not implemented.')

    def predict(self, X_featurized, *, binarized=True, **kwargs):
        Y_predict_binarized = super(LabelsClassifierMixin, self).predict(X_featurized, **kwargs)
        if binarized: return Y_predict_binarized

        return self.unbinarize_labels(Y_predict_binarized)
    #end def

    def predict_and_proba(self, X_featurized, *, binarized=True, **kwargs):
        Y_proba, Y_predict_binarized = super(LabelsClassifierMixin, self).predict_and_proba(X_featurized, **kwargs)
        if binarized: return Y_proba, Y_predict_binarized

        return Y_proba, self.unbinarize_labels(Y_predict_binarized)
    #end def

    def binarize_labels(self, Y_labels, **kwargs): raise NotImplementedError('binarize_labels is not implemented.')

    def binarize_dicts(self, Y_dicts, *, default=0.0, **kwargs): raise NotImplementedError('binarize_dicts is not implemented.')

    def unbinarize_labels(self, Y_binarized, *, epsilon=0.0, **kwargs): raise NotImplementedError('unbinarize_labels is not implemented.')

    def _filter_labels(self, Y_labels):
        if not self.exclude and not self.include: return Y_labels
        if self.include: logger.debug('Included labels: {}'.format(', '.join(self.include)))
        if self.exclude: logger.debug('Excluded labels: {}'.format(', '.join(self.exclude)))

        Y_labels_filtered = np.empty(Y_labels.shape, dtype=np.object)
        removed_labels = 0
        for i in range(Y_labels.shape[0]):
            Y_labels_filtered[i] = [l for l in Y_labels[i] if (l in self.include or not self.include) and l not in self.exclude]
            removed_labels += len(Y_labels[i]) - len(Y_labels_filtered[i])
        #end for
        if removed_labels: logger.info('{} label-instances removed from the training data.'.format(removed_labels))

        return Y_labels_filtered
    #end def
#end def


class BinaryLabelsClassifier(LabelsClassifierMixin):
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

    def binarize_labels(self, Y_labels, **kwargs):
        Y_binarized = np.zeros((Y_labels.shape[0], 2))
        for i in range(Y_labels.shape[0]):
            if self.pos_label in Y_labels[i]: Y_binarized[i, 1] = 1
            else: Y_binarized[i, 0] = 1
        #end for

        return Y_binarized
    #end def

    def binarize_dicts(self, Y_dicts, *, default=0.0, **kwargs):
        Y_binarized = np.zeros((Y_dicts.shape[0], 2))
        for i in range(Y_dicts.shape[0]):
            Y_binarized[i, 1] = Y_dicts[i].get(self.pos_label, default)
        Y_binarized[:, 0] = 1.0 - Y_binarized[:, 1]

        return Y_binarized
    #end def

    def unbinarize_labels(self, Y_proba, *, epsilon=0.0, to_dict=False, **kwargs):
        if Y_proba.ndim == 2: Y_proba = Y_proba[:, 1]

        unbinarized = np.empty(Y_proba.shape[0], dtype=np.object)
        for i in range(Y_proba.shape[0]):
            if to_dict: unbinarized[i] = dict([(self.classes_[0], 1.0 - float(Y_proba[i])), (self.classes_[1], float(Y_proba[i]))])
            else: unbinarized[i] = [self.pos_label if Y_proba[i] > 0.5 else self.not_pos_label]
        #end for

        return unbinarized
    #end def

    @property
    def classes_(self):
        return [self.not_pos_label, self.pos_label]
#end def


class MultiLabelsClassifier(LabelsClassifierMixin):
    def _fit(self, X, Y_labels, **kwargs):
        Y_labels_filtered = self._filter_labels(Y_labels)
        self.label_binarizer_ = MultiLabelBinarizer(sparse_output=False).fit(Y_labels_filtered)
        logger.info('{} labels found in training instances.'.format(len(self.classes_)))

        if not len(self.classes_): raise ValueError('There are no labels available for fitting model.')

        return super(MultiLabelsClassifier, self)._fit(X, Y_labels_filtered, **kwargs)
    #end def

    def binarize_labels(self, Y_labels, **kwargs):
        classes_ = set(self.classes_)

        return self.label_binarizer_.transform((filter(classes_.__contains__, labels) for labels in Y_labels))
    #end def

    def binarize_dicts(self, Y_dicts, *, default=0.0, **kwargs):
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

    def unbinarize_labels(self, Y_proba, *, epsilon=0.0, to_dict=False, **kwargs):
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


class MulticlassLabelsClassifier(MultiLabelsClassifier):
    def _fit(self, X, Y_labels, **kwargs):
        Y_labels_filtered = self._filter_labels(Y_labels)
        if any(len(Y_labels_filtered[i]) > 1 for i in range(Y_labels_filtered.shape[0])):
            logger.warning('Some of Y_labels contain more than 1 labels but this is a multiclass classifier. Only the first labels will be used.')
        Y_labels_filtered = np.array([Y_labels_filtered[i][0] if Y_labels_filtered[i] else '<none>' for i in range(Y_labels_filtered.shape[0])])

        return super(MulticlassLabelsClassifier, self)._fit(X, Y_labels_filtered, **kwargs)
    #end def

    def binarize_labels(self, Y_labels, **kwargs):
        if any(len(Y_labels[i]) > 1 for i in range(Y_labels.shape[0])):
            logger.warning('Some of Y_labels contain more than 1 labels but this is a multiclass classifier. Only the first labels will be used.')

        Y_labels_filtered = np.array([Y_labels[i][0] if Y_labels[i] else '<none>' for i in range(Y_labels.shape[0])])

        return super(MulticlassLabelsClassifier, self)._fit(Y_labels_filtered, **kwargs)
    #end def
#end def
