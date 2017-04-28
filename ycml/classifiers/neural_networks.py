import logging
import os
import shutil
from tempfile import NamedTemporaryFile
import time

try:
    from keras import backend as K
    from keras.callbacks import Callback, ModelCheckpoint
    from keras.models import load_model

    import tensorflow as tf
except ImportError:
    Callback = object

import numpy as np


__all__ = ['KerasNNClassifierMixin', 'keras_f1_score', 'EarlyStopping']

logger = logging.getLogger(__name__)


class KerasNNClassifierMixin(object):
    def __init__(self, tf_config=None, validation_size=0.1, epochs=10, batch_size=128, log_device_placement=False, verbose=0, early_stopping=None, save_best=None, save_weights=None, **kwargs):
        self.tf_config = tf_config
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_device_placement = log_device_placement
        self.validation_size = validation_size
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.save_weights = save_weights
        self.save_best = save_best

        self.set_session(tf_config)
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
        with NamedTemporaryFile(prefix=__name__ + '.', suffix='.h5', delete=True) as f:
            self.nn_model_.save(f.name)
            tar_file.add(f.name, arcname='nn_model.h5')
        #end with

        return self
    #end def

    def load_from_tarfile(self, tar_file):
        self.set_session()

        with NamedTemporaryFile(prefix='ix_aols_issues.', suffix='.h5', delete=True) as f:
            start_time = time.time()
            shutil.copyfileobj(tar_file.extractfile('nn_model.h5'), f)
            self.nn_model_ = load_model(f.name, custom_objects=self.custom_objects)
            logger.debug('Loaded neural network model weights (took {:.3f} seconds).'.format(time.time() - start_time))
        #end with

        return self
    #end def

    def __getstate__(self):
        return dict((k, v) for k, v in self.__dict__.items() if k not in ['nn_model_', 'tf_session'])
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
        val_keras_f1_score = logs.get(self.monitor, 0.0)
        self.val_scores.append(val_keras_f1_score)
        if len(self.val_scores) < self.patience or epoch <= self.min_epoch or val_keras_f1_score < 0.2:  # hard limit
            return

        if self.min_epoch == epoch + 1:
            logger.info('Epoch {} > {} (min_epoch). Starting early stopping monitor.'.format(epoch, self.min_epoch))

        m = np.mean(self.val_scores[-self.patience - 1:-1])
        delta = val_keras_f1_score - m
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
