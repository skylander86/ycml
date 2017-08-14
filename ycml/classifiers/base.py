__all__ = ['BaseClassifier', 'load_classifier']

from io import BytesIO
from datetime import datetime
import logging
import pickle
import tarfile
import time
from uuid import uuid4

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from ..utils import Timer
from ..utils import parse_n_jobs

logger = logging.getLogger(__name__)


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=1, **kwargs):
        super(BaseClassifier, self).__init__()

        self.n_jobs = parse_n_jobs(n_jobs)
        self.n_jobs_string = n_jobs
    #end def

    def fit(self, X, Y, validation_data=None, **kwargs):
        self.uuid_ = str(uuid4())
        logger.debug('{} UUID is {}.'.format(self.name, self.uuid_))

        self.fitted_at_ = datetime.utcnow()

        timer = Timer()
        self._fit(X, Y, validation_data=validation_data, **kwargs)
        logger.info('{} fitting on {} instances complete {}.'.format(self.name, X.shape[0], timer))

        return self
    #end def

    def _fit(self, *args, **kwargs): raise NotImplementedError('_fit is not implemented.')

    def _predict_proba(self, X_featurized, **kwargs):
        raise NotImplementedError('_predict_proba is not implemented.')

    def predict_proba(self, X_featurized, **kwargs):
        timer = Timer()
        Y_proba = self._predict_proba(X_featurized, **kwargs)
        logger.debug('Computed prediction probabilities on {} instances {}.'.format(X_featurized.shape[0], timer))

        return Y_proba
    #end def

    def predict(self, X_featurized, **kwargs):
        timer = Timer()
        Y_proba = self._predict_proba(X_featurized, **kwargs)
        Y_predict = Y_proba >= 0.5
        logger.debug('Computed predictions on {} instances {}.'.format(X_featurized.shape[0], timer))

        return Y_predict
    #end def

    def predict_and_proba(self, X_featurized, **kwargs):
        timer = Timer()
        Y_proba = self._predict_proba(X_featurized, **kwargs)
        Y_predict = Y_proba >= 0.5
        logger.debug('Computed predictions and probabilities on {} instances {}.'.format(X_featurized.shape[0], timer))

        return Y_proba, Y_predict
    #end def

    def decision_function(self, *args, **kwargs): return self.predict_proba(*args, **kwargs)

    def save(self, f):
        if not hasattr(self, 'uuid_'):
            raise NotFittedError('This featurizer is not fitted yet.')

        with tarfile.open(fileobj=f, mode='w') as tf:
            with BytesIO() as model_f:
                try: pickle.dump(self, model_f, protocol=4)
                except pickle.PicklingError:
                    logger.error('PicklingError: Did you check to make sure that the classifier mixins (i.e., KerasNNClassifierMixin) is ahead of BaseClassifier in the MRO?')
                    raise
                #end try

                model_data = model_f.getvalue()
                model_f.seek(0)
                model_tarinfo = tarfile.TarInfo(name='model.pkl')
                model_tarinfo.size = len(model_data)
                model_tarinfo.mtime = int(time.time())
                tf.addfile(tarinfo=model_tarinfo, fileobj=model_f)
            #end with

            self.save_to_tarfile(tf)
        #end with

        f.close()

        logger.info('{} saved to <{}>.'.format(self, f.name))

        return self
    #end def

    def save_to_tarfile(self, tf): return self

    def load_from_tarfile(self, tf): return self

    @property
    def uuid(self):
        return self.uuid_
    #end def

    @property
    def name(self):
        return type(self).__name__

    @property
    def classes_(self): raise NotImplementedError('classes_ is not implemented.')

    def __str__(self):
        return '{}(UUID={})'.format(self.name, self.uuid_ if hasattr(self, 'uuid_') else 'None')
#end class


def load_classifier(f):
    with tarfile.open(fileobj=f, mode='r') as tf:
        classifier = pickle.load(tf.extractfile('model.pkl'))
        classifier.load_from_tarfile(tf)
    #end with

    logger.info('Loaded {} from <{}>.'.format(classifier, f.name))

    return classifier
#end def
