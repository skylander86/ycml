import numpy as np

from sklearn.svm import SVC

from ycml.classifiers import BinaryLabelsClassifier, MultiLabelsClassifier

__all__ = ['HelloWorldBinaryLabelsClassifier', 'HelloWorldMultiLabelsClassifier', 'CLASSIFERS_MAP']


class HelloWorldBinaryLabelsClassifier(BinaryLabelsClassifier):
    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        self.classifier_ = SVC(probability=True).fit(X_featurized, Y_binarized)

        return self
    #end def

    def _predict_proba(self, X_featurized, **kwargs):
        return self.classifier_.predict_proba(X_featurized, **kwargs)
#end class


class HelloWorldMultiLabelsClassifier(MultiLabelsClassifier):
    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        self.classifiers_ = {}
        for j, c in enumerate(self.classes_):
            self.classifiers_[c] = SVC(probability=True).fit(X_featurized, Y_binarized[:, j])

        return self
    #end def

    def _predict_proba(self, X_featurized, **kwargs):
        Y_proba = np.zeros((X_featurized.shape[0], len(self.classes_)), dtype=np.float)

        for j, c in enumerate(self.classes_):
            Y_proba[:, j] = self.classifiers_[c].predict_proba(X_featurized, **kwargs)[:, 1]

        return Y_proba
    #end def
#end class


CLASSIFERS_MAP = {
    'HelloWorldBinaryLabelsClassifier': HelloWorldBinaryLabelsClassifier,
    'HelloWorldMultiLabelsClassifier': HelloWorldMultiLabelsClassifier,
}
