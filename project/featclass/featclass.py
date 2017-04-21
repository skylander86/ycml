from argparse import ArgumentParser
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from ..classifiers import load_classifier, get_thresholds_from_file
from ..featurizers import load_featurizer

from ..utils.uriutils import uri_open

__all__ = ['FeaturizerAndClassifier']

logger = logging.getLogger(__name__)


class FeaturizerAndClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, featurizer_uri=None, classifier_uri=None, thresholds_uri=None, **kwargs):
        self.featurizer_uri = featurizer_uri
        self.classifier_uri = classifier_uri
        self.thresholds_uri = thresholds_uri

        if not self.featurizer_uri: raise Exception('Featurizer URI is not set!')
        if not self.classifier_uri: raise Exception('Classifier URI is not set!')

        with uri_open(self.featurizer_uri, 'rb') as f:
            self.featurizer_ = load_featurizer(f)

        with uri_open(self.classifier_uri, 'rb') as f:
            self.classifier_ = load_classifier(f)

        self.thresholds_ = np.full(len(self.classifier_.classes_), 0.5, dtype=np.float)
        if self.thresholds_uri:
            with uri_open(self.thresholds_uri) as f:
                self.thresholds_ = get_thresholds_from_file(f, self.classifier_.classes_)

        self.denominators_ = 2.0 * (1.0 - self.thresholds_)

        if 'featurizer_uuid' in kwargs and self.featurizer_.uuid != kwargs['featurizer_uuid']:
            raise TypeError('Featurizer UUID mismatch: {} (this) != {} (<{}>)'.format(kwargs['featurizer_uuid'], self.featurizer_.uuid, self.featurizer_uri))
        elif 'classifier_uuid' in kwargs and self.classifier_.uuid != kwargs['classifier_uuid']:
            raise TypeError('Classifier UUID mismatch: {} (this) != {} (<{}>)'.format(kwargs['classifier_uuid'], self.classifier_.uuid, self.classifier_uri))
    #end def

    def fit(self, *args, **kwargs):
        raise NotImplementedError('FeaturizerAndClassifier does not support the `fit` method.')

    def transform(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)
    #end def

    def predict(self, X, binarized=False, **kwargs):
        X_featurized = self.featurizer_.transform(X)
        Y_predict_binarized = self.classifier_.predict(X_featurized, thresholds=self.thresholds_, **kwargs)
        if binarized: return Y_predict_binarized

        return self.classifier_.unbinarize_labels(Y_predict_binarized)
    #end def

    def predict_and_proba(self, X, **kwargs):
        X_featurized = self.featurizer_.transform(X)
        return self.classifier_.predict_and_proba(X_featurized, self.thresholds_)
    #end def

    def predict_proba(self, X, rescale=False, **kwargs):
        X_featurized = self.featurizer_.transform(X)
        Y_proba = self.classifier_.predict_proba(X_featurized, **kwargs)

        if rescale: return self.rescale_proba(Y_proba)

        return Y_proba
    #end def

    def _rescale_proba(self, Y_proba):
        n_classes = len(self.classifier_.classes_)

        rescaled = np.zeros(Y_proba.shape)
        for i in range(Y_proba.shape[0]):
            for k in range(n_classes):
                if Y_proba[i, k] > self.thresholds_[k]: rescaled[i, k] = 0.5 + ((Y_proba[i, k] - self.thresholds_[k]) / self.denominators_[k])
                else: rescaled[i, k] = Y_proba[i, k] / (self.thresholds_[k] + self.thresholds_[k])
            #end for
        #end for

        return rescaled
    #end def

    def decision_function(self, *args, **kwargs):
        return self.predict_proba(*args, **kwargs)

    def __str__(self):
        return 'FeaturizerAndClassifier(featurizer_uri={}, classifier_uri={}, thresholds_uri={})'.format(self.featurizer_uri, self.classifier_uri, self.thresholds_uri)
    #end def
#end class


def main():
    parser = ArgumentParser(description='Loads a featurizer-classifier. (DEVELOPMENT SCRIPT)')
    parser.add_argument('featclass_uri', type=str, metavar='<featclass_uri>', help='Featclass settings file to load.')
    A = parser.parse_args()

    logging.basicConfig(format=u'%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    featclass = FeaturizerAndClassifier(featclass_uri=A.featclass_uri)
    print(featclass.transform(np.array([dict(content='this is a tort case. responsibility')]), verbose=0))
    print(featclass.predict_proba(np.array([dict(content='this is a tort case. responsibility')]), rescale=True, verbose=0))
#end def


if __name__ == '__main__': main()
