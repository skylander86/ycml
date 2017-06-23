from argparse import ArgumentParser
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from ..classifiers import load_classifier
from ..featurizers import load_featurizer

from ..utils import uri_open, get_settings

__all__ = ['BaseFeatClass']

logger = logging.getLogger(__name__)


class BaseFeatClass(BaseEstimator, ClassifierMixin):
    def __init__(self, settings={}, check_environment=True, featurizer_uri=None, classifier_uri=None, **kwargs):
        super(BaseFeatClass, self).__init__(**kwargs)

        sources = ('env', settings) if check_environment else (settings,)

        self.featurizer_uri = get_settings(key='featurizer_uri', sources=sources, raise_on_missing=True) if featurizer_uri is None else featurizer_uri
        self.classifier_uri = get_settings(key='classifier_uri', sources=sources, raise_on_missing=True) if classifier_uri is None else classifier_uri

        with uri_open(self.featurizer_uri, 'rb') as f:
            self.featurizer_ = load_featurizer(f)

        with uri_open(self.classifier_uri, 'rb') as f:
            self.classifier_ = load_classifier(f)

        if 'featurizer_uuid' in kwargs and self.featurizer_.uuid != kwargs['featurizer_uuid']:
            raise TypeError('Featurizer UUID mismatch: {} (this) != {} (<{}>)'.format(kwargs['featurizer_uuid'], self.featurizer_.uuid, self.featurizer_uri))
        elif 'classifier_uuid' in kwargs and self.classifier_.uuid != kwargs['classifier_uuid']:
            raise TypeError('Classifier UUID mismatch: {} (this) != {} (<{}>)'.format(kwargs['classifier_uuid'], self.classifier_.uuid, self.classifier_uri))
    #end def

    def fit(self, *args, **kwargs):
        raise NotImplementedError('BaseFeatClass does not support the `fit` method.')

    def transform(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)
    #end def

    def predict(self, X, **kwargs):
        X_featurized = self.featurizer_.transform(X, **kwargs)

        return self.classifier_.predict(X_featurized, **kwargs)
    #end def

    def predict_proba(self, X, **kwargs):
        X_featurized = self.featurizer_.transform(X, **kwargs)

        return self.classifier_.predict_proba(X_featurized, **kwargs)
    #end def

    def predict_and_proba(self, X, **kwargs):
        X_featurized = self.featurizer_.transform(X, **kwargs)

        return self.classifier_.predict_and_proba(X_featurized, **kwargs)
    #end def

    def decision_function(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)

    def __str__(self):
        return 'BaseFeatClass(featurizer_uri={}, classifier_uri={}, thresholds_uri={})'.format(self.featurizer_uri, self.classifier_uri, self.thresholds_uri)
    #end def
#end class


def main():
    parser = ArgumentParser(description='Loads a featurizer-classifier. (DEVELOPMENT SCRIPT)')
    parser.add_argument('featclass_uri', type=str, metavar='<featclass_uri>', help='Featclass settings file to load.')
    A = parser.parse_args()

    logging.basicConfig(format=u'%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    featclass = BaseFeatClass(featclass_uri=A.featclass_uri)
    print(featclass.transform(np.array([dict(content='this is a tort case. responsibility')]), verbose=0))
    print(featclass.predict_proba(np.array([dict(content='this is a tort case. responsibility')]), rescale=True, verbose=0))
#end def


if __name__ == '__main__': main()
