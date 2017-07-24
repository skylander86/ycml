__all__ = ['KerasNNClassifierMixin', 'keras_f1_score', 'EarlyStopping']

import logging
import math
import os
import shutil
from tempfile import NamedTemporaryFile

try:
    from keras import backend as K
    from keras.callbacks import Callback, ModelCheckpoint
    from keras.models import load_model

    import tensorflow as tf
except ImportError:
    Callback = object

import numpy as np
import scipy.sparse as sps

from sklearn.model_selection import train_test_split

from uriutils import uri_open

from ..utils import Timer

logger = logging.getLogger(__name__)


class KerasNNClassifierMixin(object):
    PICKLE_IGNORED_ATTRIBUTES = set()
    NN_MODEL_ATTRIBUTE = 'nn_model_'

    def __init__(
        self,
        tf_config=None, set_session=True,  # set_session should be False when you are initializing a second classifier...
        epochs=10, batch_size=128, passes_per_epoch=1,
        initial_weights=None, initial_epoch=0,
        validation_size=0.2, verbose=0,
        early_stopping=None, save_best=None, save_weights=None,
        log_device_placement=False,
        **kwargs
    ):
        self.tf_config = tf_config

        self.epochs = epochs
        self.batch_size = batch_size
        self.passes_per_epoch = passes_per_epoch

        self.initial_weights = initial_weights
        self.initial_epoch = initial_epoch

        self.validation_size = validation_size
        self.verbose = verbose

        self.early_stopping = early_stopping
        self.save_weights = save_weights
        self.save_best = save_best

        self.log_device_placement = log_device_placement

        if set_session: self.set_session(tf_config)
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        setattr(self, self.NN_MODEL_ATTRIBUTE, self.compile_model(**kwargs))
        self.keras_fit_generator(X_featurized, Y_binarized, **kwargs)

        return self
    #end def

    def compile_model(self, X_featurized, Y_binarized, **kwargs): raise NotImplementedError('compile_model is not implemented.')

    def _predict_proba(self, X_featurized, batch_size=1024, use_generator=False, **kwargs):
        if X_featurized.shape[0] < batch_size: kwargs['verbose'] = 0

        if use_generator or X_featurized.shape[0] > batch_size: Y_proba = self.keras_predict_generator(X_featurized, batch_size=batch_size, **kwargs)
        else: Y_proba = self.keras_predict(X_featurized, batch_size=batch_size, **kwargs)

        return Y_proba
    #end def

    def set_session(self, tf_config=None):
        if tf_config is None:
            tf_config = self.tf_config

        if tf_config is None:
            n_jobs = getattr(self, 'n_jobs', 1)
            log_device_placement = getattr(self, 'log_device_placement', logger.getEffectiveLevel() <= logging.DEBUG)
            tf_config = tf.ConfigProto(inter_op_parallelism_threads=n_jobs, intra_op_parallelism_threads=n_jobs, log_device_placement=log_device_placement, allow_soft_placement=True)
        #end if

        tf_session = tf.Session(config=tf_config)
        K.set_session(tf_session)

        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)
    #end def

    def keras_fit(self, X, Y, *, nn_model=None, validation_data=None, resume=None, **fit_args):
        if nn_model is None: nn_model = getattr(self, self.NN_MODEL_ATTRIBUTE)

        if not self._pre_fit_setup(nn_model, resume=resume, **fit_args): return

        if sps.issparse(X): X = X.toarray()
        if sps.issparse(Y): Y = Y.toarray()
        if validation_data is not None:
            X_validation, Y_validation = validation_data
            validation_data = (X_validation.toarray() if sps.issparse(X_validation) else X_validation, Y_validation.toarray() if sps.issparse(Y_validation) else Y_validation)
        #end if

        logger.info('{} instances used for training and {} instances used for validation.'.format(Y.shape[0], validation_data[1].shape[0] if validation_data else int(self.validation_size * Y.shape[0])))

        return nn_model.fit(X, Y, validation_data=validation_data, validation_split=0.0 if validation_data is not None else self.validation_size, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=self.build_callbacks(), initial_epoch=self.initial_epoch, **fit_args)
    #end def

    def keras_fit_generator(self, X, Y, *, nn_model=None, generator_func=None, validation_data=None, resume=None, **fit_args):
        if nn_model is None: nn_model = getattr(self, self.NN_MODEL_ATTRIBUTE)

        if not self._pre_fit_setup(nn_model, resume=resume, **fit_args): return

        if validation_data is None:
            X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=self.validation_size)
            validation_data = (X_validation.toarray() if sps.issparse(X_validation) else X_validation, Y_validation.toarray() if sps.issparse(Y_validation) else Y_validation)
        else:
            X_train, Y_train = X, Y
            X_validation, Y_validation = validation_data
            validation_data = (X_validation.toarray() if sps.issparse(X_validation) else X_validation, Y_validation.toarray() if sps.issparse(Y_validation) else Y_validation)
        #end if
        N_train = X_train.shape[0]
        logger.info('{} instances used for training and {} instances used for validation.'.format(N_train, validation_data[1].shape[0]))

        steps_per_epoch = int((N_train * self.passes_per_epoch) / self.batch_size)
        if steps_per_epoch <= 0: raise ValueError('steps_per_epoch ({}) is <= 0!'.format(steps_per_epoch))
        logger.info('Fit generator will run {} steps per epoch with batch size of {}. This will make 1 pass through the training data in {:.2f} epochs.'.format(steps_per_epoch, self.batch_size, N_train / (steps_per_epoch * self.batch_size)))

        if generator_func is None: generator_func = self._generator

        return nn_model.fit_generator(
            generator_func(X_train, Y_train, batch_size=self.batch_size),
            validation_data=validation_data,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=steps_per_epoch, epochs=self.epochs, verbose=self.verbose, callbacks=self.build_callbacks(),
            **fit_args
        )
    #end def

    def _pre_fit_setup(self, nn_model, *, resume=None, **kwargs):
        if resume:
            self.initial_weights = resume[0]

            if len(resume) > 1:
                self.initial_epoch = int(resume[1])
                if self.initial_epoch < 0:
                    logger.info('Detected negative initial epoch value. Will not perform model fitting.')
                    self.epochs = 0
                #end if
            #end if
        #end if

        if self.initial_weights:
            with uri_open(self.initial_weights, in_memory=False) as f:
                nn_model.load_weights(f.temp_name)
            logger.info('Loaded initial weights file from <{}> will start at epoch {}.'.format(self.initial_weights, self.initial_epoch))
        #end if

        if self.epochs == 0:
            logger.warning('Epochs is set to 0. Model fitting will not continue.')
            return
        #end if

        return True
    #end def

    def _generator(self, X, Y, *, batch_size=128, shuffle=True):
        N = X.shape[0]
        if batch_size > N: raise ValueError('batch_size ({}) is > than number of instances ({}).'.format(batch_size, N))

        sparse_X = sps.issparse(X)
        if Y is None: batch_sized_zeros = np.zeros(batch_size)

        if shuffle:
            shuffled_indexes = np.random.permutation(N)
            X_shuffled = X[shuffled_indexes, ...]
            if Y is not None: Y_shuffled = Y[shuffled_indexes]
        else: X_shuffled, Y_shuffled = X, Y

        cur = 0
        while True:
            if cur + batch_size > N:
                if shuffle:
                    shuffled_indexes = np.random.permutation(N)
                    X_shuffled = X[shuffled_indexes, :]
                    if Y is not None: Y_shuffled = Y[shuffled_indexes]
                #end if

                cur = 0
            #end if

            X_batch = X_shuffled[cur:cur + batch_size, ...].toarray() if sparse_X else X_shuffled[cur:cur + batch_size, ...]
            Y_batch = batch_sized_zeros if Y is None else Y_shuffled[cur:cur + batch_size]

            yield (X_batch, Y_batch)

            cur += batch_size
        #end while
    #end def

    def keras_predict(self, X, *, nn_model=None, **kwargs):
        if nn_model is None: nn_model = getattr(self, self.NN_MODEL_ATTRIBUTE)

        batch_size = kwargs.pop('batch_size', self.batch_size)
        verbose = kwargs.pop('verbose', self.verbose)

        return nn_model.predict_proba(X.toarray() if sps.issparse(X) else X, batch_size=batch_size, verbose=verbose)
    #end def

    def keras_predict_generator(self, X, *, nn_model=None, **kwargs):
        if nn_model is None: nn_model = getattr(self, self.NN_MODEL_ATTRIBUTE)

        N = X.shape[0]

        batch_size = kwargs.pop('batch_size', self.batch_size)
        verbose = kwargs.pop('verbose', self.verbose)
        steps = int(math.ceil(N / batch_size))

        return nn_model.predict_generator(self._generator(X, None, batch_size=batch_size, shuffle=False), steps=steps, verbose=verbose)[:N, ...]
    #end def

    def build_callbacks(self):
        callbacks = []
        if self.early_stopping is not None:
            early_stopping = EarlyStopping(**self.early_stopping)
            callbacks.append(early_stopping)
            logger.info('Set up {}.'.format(early_stopping))
        #end if

        if self.save_best is not None:
            monitor = self.save_best.get('monitor', 'accuracy')
            filepath = self.save_best.get('path', '{epoch:04d}_{' + monitor + ':.5f}.h5')
            callbacks.append(ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=self.verbose, save_best_only=True, save_weights_only=True, mode='max'))
            logger.info('Best `{}` model will be saved to <{}>.'.format(monitor, filepath))
        #end if

        if self.save_weights is not None:
            path = self.save_weights.get('path', '.')
            period = self.save_weights.get('period', 1)
            filepath = os.path.join(self.save_weights, '{epoch:04d}_{val_keras_f1_score:.5f}.h5') if os.path.isdir(path) else path
            callbacks.append(ModelCheckpoint(filepath=filepath, verbose=self.verbose, save_best_only=False, save_weights_only=True, period=period))
            logger.info('Weights of every {}th model will be saved to <{}>.'.format(period, filepath))
        #end if

        return callbacks
    #end def

    def save_to_tarfile(self, tar_file):
        with NamedTemporaryFile(prefix=__name__ + '.', suffix='.h5', delete=False) as f:
            temp_path = f.name
        #end with

        self.nn_model_.save(temp_path)
        tar_file.add(temp_path, arcname='nn_model.h5')
        os.remove(temp_path)

        return self
    #end def

    def load_from_tarfile(self, tar_file):
        self.set_session()

        fname = None
        try:
            with NamedTemporaryFile(suffix='.h5', delete=False) as f:
                timer = Timer()
                shutil.copyfileobj(tar_file.extractfile('nn_model.h5'), f)
                fname = f.name
            #end with

            self.nn_model_ = load_model(fname, custom_objects=self.custom_objects)
            logger.debug('Loaded neural network model weights {}.'.format(timer))

        finally:
            if fname:
                os.remove(fname)
        #end try

        return self
    #end def

    def __getstate__(self):
        state = super(KerasNNClassifierMixin, self).__getstate__()

        ignored_attrs = set([self.NN_MODEL_ATTRIBUTE, 'tf_session']) | self.PICKLE_IGNORED_ATTRIBUTES
        for k in ignored_attrs:
            if k in state:
                del state[k]

        return state
    #end def

    @property
    def custom_objects(self): return {}
#end class


def keras_f1_score(y_true, y_pred):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    # If there are no true positives, fix the F score at 0 like sklearn.

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    beta = 1.0

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())

    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score
#end def


class EarlyStopping(Callback):
    def __init__(self, monitor='val_accuracy', patience=5, min_delta=0.01, min_epoch=50):
        super(EarlyStopping, self).__init__()

        self.patience = patience
        self.min_delta = min_delta
        self.min_epoch = min_epoch
        self.monitor = monitor

        self.val_scores = []
    #end def

    def on_epoch_end(self, epoch, logs={}):
        monitor_score = logs[self.monitor]
        self.val_scores.append(monitor_score)
        if len(self.val_scores) < self.patience or epoch <= self.min_epoch or monitor_score < 0.2:  # hard limit
            return

        if self.min_epoch == epoch + 1:
            logger.info('Epoch {} > {} (min_epoch). Starting early stopping monitor.'.format(epoch, self.min_epoch))

        m = np.mean(self.val_scores[-self.patience - 1:-1])
        delta = abs(monitor_score - m)
        min_delta = self.min_delta * m
        # logger.debug('mean({}[-{}:])={}; delta={}; min_delta={};'.format(self.monitor, self.patience, m, delta, min_delta))

        if delta < min_delta:
            logger.info('Model delta fell below `min_delta` threshold. Early stopped.')
            self.model.stop_training = True
        #end if
    #end def

    def __str__(self):
        return 'EarlyStopping(monitor={}, patience={}, min_delta={}, min_epoch={})'.format(self.monitor, self.patience, self.min_delta, self.min_epoch)
#end def
