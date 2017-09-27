__all__ = ['BaseFeatClass', 'load_featclass']

from argparse import ArgumentParser
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from ycsettings import Settings

from ..classifiers import load_classifier
from ..featurizers import load_featurizer

from uriutils import uri_open

from ..utils import get_class_from_module_path, chunked_iterator

logger = logging.getLogger(__name__)


class BaseFeatClass(BaseEstimator, ClassifierMixin):
    def __init__(self, featurizer_uri, classifier_uri, *, transform_args={}, predict_args={}, **kwargs):
        super(BaseFeatClass, self).__init__(**kwargs)

        self.featurizer_uri = featurizer_uri
        self.classifier_uri = classifier_uri

        self.transform_args = transform_args
        self.predict_args = predict_args

        with uri_open(featurizer_uri, mode='rb') as f:
            self.featurizer = load_featurizer(f)

        with uri_open(classifier_uri, mode='rb') as f:
            self.classifier = load_classifier(f)

        if 'featurizer_uuid' in kwargs and self.featurizer.uuid != kwargs['featurizer_uuid']:
            raise TypeError('Featurizer UUID mismatch: {} (this) != {} (<{}>)'.format(kwargs['featurizer_uuid'], self.featurizer.uuid, featurizer_uri))
        elif 'classifier_uuid' in kwargs and self.classifier.uuid != kwargs['classifier_uuid']:
            raise TypeError('Classifier UUID mismatch: {} (this) != {} (<{}>)'.format(kwargs['classifier_uuid'], self.classifier.uuid, classifier_uri))
    #end def

    def fit(self, *args, **kwargs):
        raise NotImplementedError('BaseFeatClass does not support the `fit` method.')

    def transform(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)
    #end def

    def predict(self, X, **kwargs):
        predict_args = self.predict_args.copy()
        predict_args.update(kwargs)

        X_featurized = self.featurizer.transform(X, **self.transform_args)

        return self.classifier.predict(X_featurized, **predict_args)
    #end def

    def predict_proba(self, X, **kwargs):
        predict_args = self.predict_args.copy()
        predict_args.update(kwargs)

        X_featurized = self.featurizer.transform(X, **self.transform_args)

        return self.classifier.predict_proba(X_featurized, **predict_args)
    #end def

    def predict_and_proba(self, X, **kwargs):
        predict_args = self.predict_args.copy()
        predict_args.update(kwargs)

        X_featurized = self.featurizer.transform(X, **self.transform_args)

        return self.classifier.predict_and_proba(X_featurized, **predict_args)
    #end def

    def predictions_generator(self, instances_generator, *, chunk_size=100000, include_proba=True, unbinarized=True):
        for chunk in chunked_iterator(instances_generator, chunk_size):
            X = np.array(chunk, dtype=np.object)

            if include_proba:
                Y_proba, Y_predict = self.predict_and_proba(X)
                Y_proba_dicts = self.classifier.unbinarize_labels(Y_proba, to_dict=True)
            else:
                Y_predict = self.predict(X)
                Y_proba_dicts = None
            #end if

            Y_predict_list = self.classifier.unbinarize_labels(Y_predict, to_dict=False)

            if include_proba:
                if unbinarized:
                    yield from ((X[i], Y_predict_list[i], Y_proba_dicts[i]) for i in range(X.shape[0]))
                else:
                    yield from ((X[i], Y_predict[i], Y_proba[i]) for i in range(X.shape[0]))
            else:
                if unbinarized:
                    yield from ((X[i], Y_predict_list[i]) for i in range(X.shape[0]))
                else:
                    yield from ((X[i], Y_predict[i]) for i in range(X.shape[0]))
            #end if
        #end for
    #end def

    def decision_function(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)

    def __str__(self):
        return 'BaseFeatClass(featurizer_uri={}, classifier_uri={})'.format(self.featurizer_uri, self.classifier_uri)
    #end def
#end class


def load_featclass(*, settings=None, uri=None, check_environment=True):
    if settings is None:
        settings = Settings(uri)

    if not isinstance(settings, Settings):
        settings = Settings(settings, uri)

    featclass_type = settings.get('featclass_type', raise_exception=True)
    featclass_parameters = settings.getdict('featclass_parameters', default={})

    featclass_class = get_class_from_module_path(featclass_type)
    featclass = featclass_class(**featclass_parameters)

    logger.info('Loaded {} from <{}>.'.format(featclass, uri if uri else ('env/settings' if check_environment else 'settings')))

    return featclass
#end def


def main():
    parser = ArgumentParser(description='Loads a featurizer-classifier. (DEVELOPMENT SCRIPT)')
    parser.add_argument('featclass_uri', type=str, metavar='<featclass_uri>', help='Featclass settings file to load.')
    A = parser.parse_args()

    logging.basicConfig(format=u'%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    featclass = load_featclass(uri=A.featclass_uri)
    print(featclass.transform(np.array([dict(content='this is a tort case. responsibility')])))
    print(featclass.predict_proba(np.array([dict(content='this is a tort case. responsibility')])))
#end def


if __name__ == '__main__': main()
