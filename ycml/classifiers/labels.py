__all__ = ['BinaryLabelsClassifier', 'MultiLabelsClassifier', 'MulticlassLabelsClassifier', 'filter_labels']

import logging

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseClassifier

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

    def binarize_labels(self, Y_labels, **kwargs): raise NotImplementedError('binarize_labels is not implemented.')

    def binarize_dicts(self, Y_dicts, *, default=0.0, **kwargs): raise NotImplementedError('binarize_dicts is not implemented.')

    def unbinarize_labels(self, Y_proba, *, epsilon=1e-5, to_dict=False, astype=float, **kwargs):
        assert len(self.classes_) == Y_proba.shape[1]
        unbinarized = np.empty(Y_proba.shape[0], dtype=np.object)
        for i in range(Y_proba.shape[0]):
            if to_dict: unbinarized[i] = dict((self.classes_[j], astype(Y_proba[i, j])) for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon)
            else: unbinarized[i] = [self.classes_[j] for j in range(Y_proba.shape[1]) if Y_proba[i, j] > epsilon]
        #end for

        return unbinarized
    #end def
#end def


class BinaryLabelsClassifier(LabelsClassifierMixin):
    def __init__(self, pos_label, *, not_pos_label=None, **kwargs):
        super(BinaryLabelsClassifier, self).__init__(**kwargs)

        self.pos_label = pos_label
        self.not_pos_label = 'not ' + pos_label if not_pos_label is None else not_pos_label
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

    @property
    def classes_(self):
        return [self.not_pos_label, self.pos_label]
#end def


class MultiLabelsClassifier(LabelsClassifierMixin):
    def _fit(self, X, Y_labels, **kwargs):
        Y_labels_filtered = filter_labels(Y_labels, include=self.include, exclude=self.exclude)
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
        classes_map = dict((c, i) for i, c in enumerate(self.classes_))

        for i in range(Y_dicts.shape[0]):
            d = Y_dicts[i]
            for k, p in d.items():
                try: binarized[i, classes_map[k]] = p
                except IndexError: pass
            #end for
        #end for

        return binarized
    #end def

    @property
    def classes_(self):
        return self.label_binarizer_.classes_
    #end def
#end def


class MulticlassLabelsClassifier(MultiLabelsClassifier):
    def _fit(self, X, Y_labels, **kwargs):
        Y_labels_filtered = self._filter_and_check_labels(Y_labels)
        if any(len(Y_labels_filtered[i]) == 0 for i in range(Y_labels_filtered.shape[0])):
            logger.warning('Some of Y_labels have no labels after filtering but this is a multiclass classifier. A `<none>` label will be created.')

            for i in range(Y_labels_filtered.shape[0]):
                if not Y_labels_filtered[i]:
                    Y_labels_filtered[i] = ['<none>']
        #end if

        return super(MulticlassLabelsClassifier, self)._fit(X, Y_labels_filtered, **kwargs)
    #end def

    def predict(self, X_featurized, **kwargs):
        Y_proba = self.predict_proba(X_featurized, **kwargs)
        Y_predict_binarized = np.zeros(Y_proba.shape)
        for i in range(X_featurized.shape[0]):
            j = np.argmax(Y_proba[i, :])
            Y_predict_binarized[i, j] = 1
        #end for

        return Y_predict_binarized
    #end def

    def predict_and_proba(self, X_featurized, **kwargs):
        Y_proba = super(MulticlassLabelsClassifier, self).predict_proba(X_featurized, **kwargs)

        Y_predict_binarized = np.zeros(Y_proba.shape)
        for i in range(X_featurized.shape[0]):
            j = np.argmax(Y_proba[i, :])
            Y_predict_binarized[i, j] = 1
        #end for

        return Y_proba, Y_predict_binarized
    #end def

    def binarize_labels(self, Y_labels, **kwargs):
        Y_labels_filtered = self._filter_and_check_labels(Y_labels)

        return super(MulticlassLabelsClassifier, self).binarize_labels(Y_labels_filtered, **kwargs)
    #end def

    def multiclassify_labels(self, Y_labels, **kwargs):
        Y_binarized = self.binarize_labels(Y_labels, **kwargs)
        return MulticlassLabelsClassifier.multilabel_to_multiclass(Y_binarized)
    #end def

    @classmethod
    def multilabel_to_multiclass(cls, Y_multilabel):
        Y_multiclass = np.zeros(Y_multilabel.shape[0])
        for j in range(Y_multilabel.shape[1]):
            Y_multiclass[Y_multilabel[:, j] > 0] = j

        return Y_multiclass
    #end def

    @classmethod
    def multiclass_to_multilabel(cls, Y_multiclass, n_classes=None):
        if n_classes is None: n_classes = np.max(Y_multiclass)
        Y_multilabel = np.zeros((Y_multiclass.shape[0], n_classes))
        for j in range(Y_multilabel.shape[1]):
            Y_multilabel[Y_multiclass == j, j] = 1

        return Y_multiclass
    #end def

    def _filter_and_check_labels(self, Y_labels):
        Y_labels_filtered = filter_labels(Y_labels, include=self.include, exclude=self.exclude)

        if any(len(Y_labels[i]) > 1 for i in range(Y_labels.shape[0])):
            logger.warning('Some of Y_labels contain more than 1 labels but this is a multiclass classifier. Only the first labels will be used.')

        return np.array([[Y_labels_filtered[i][0]] if Y_labels_filtered[i] else [] for i in range(Y_labels_filtered.shape[0])])
    #end def

    @property
    def classes_(self):
        return self.label_binarizer_.classes_
    #end def
#end def


def filter_labels(Y_labels, *, include=[], exclude=[]):
    if not exclude and not include: return Y_labels
    if include: logger.debug('Included labels: {}'.format(', '.join(sorted(include))))
    if exclude: logger.debug('Excluded labels: {}'.format(', '.join(sorted(exclude))))

    Y_labels_filtered = []
    removed_labels = 0
    for i in range(Y_labels.shape[0]):
        Y_labels_filtered.append([l for l in Y_labels[i] if (l in include or not include) and l not in exclude])
        removed_labels += len(Y_labels[i]) - len(Y_labels_filtered[i])
    #end for
    Y_labels_filtered = np.array(Y_labels_filtered, dtype=np.object)
    if removed_labels: logger.info('{} label-instances removed from the data.'.format(removed_labels))

    return Y_labels_filtered
#end def
