from sklearn.svm import SVC

from ycml.classifiers import BinaryLabelsClassifier

__all__ = ['HelloWorldBinaryLabelsClassifier', 'CLASSIFERS_MAP']


class HelloWorldBinaryLabelsClassifier(BinaryLabelsClassifier):
    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        self.classifier_ = SVC(probability=True).fit(X_featurized, Y_binarized)

        return self
    #end def

    def _predict_proba(self, X_featurized, **kwargs):
        return self.classifier_.predict_proba(X_featurized, **kwargs)
#end class


CLASSIFERS_MAP = {
    'HelloWorldBinaryLabelsClassifier': HelloWorldBinaryLabelsClassifier,
}
