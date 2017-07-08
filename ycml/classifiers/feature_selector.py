__all__ = ['MultiLabelsFeatureSelector']

from .labels import MultiLabelsClassifier
from ..utils import get_class_from_module_path

import numpy as np


class MultiLabelsFeatureSelector(MultiLabelsClassifier):
    def __init__(self, score_func=None, **kwargs):
        super(MultiLabelsClassifier, self).__init__(**kwargs)

        self.score_func = get_class_from_module_path(score_func)
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        scores = np.zeros((X_featurized.shape[1], Y_binarized.shape[1]))
        pval = np.zeros((X_featurized.shape[1], Y_binarized.shape[1]))
        for j in range(Y_binarized.shape[1]):
            scores[:, j], pval[:, j] = self.score_func(X_featurized, Y_binarized[:, j])

        self.labels_scores_ = scores
        self.labels_pvals_ = pval
        self.scores_ = np.max(scores, axis=1)

        return self
    #end def

    def feature_select(self, X_featurized, Y_labels, **kwargs):
        self.fit(X_featurized, Y_labels, **kwargs)

        return self.scores_
    #end def

    def transform(self, X, **kwargs):
        return X
#end class
