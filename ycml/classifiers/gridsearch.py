__all__ = ['HyperparameterGridsearchMixin']

import logging

from joblib import Parallel, delayed

from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

from ycml.utils import get_class_from_module_path

logger = logging.getLogger(__name__)


class HyperparameterGridsearchMixin(object):
    def __init__(self, classifier='sklearn.svm.SVC', classifier_args={}, param_grid={}, metric=accuracy_score, validation_size=0.2, **kwargs):
        self.classifier = classifier
        self.classifier_args = classifier_args
        self.param_grid = param_grid
        self.validation_size = validation_size

        self.metric = metric
        if isinstance(metric, str):
            if metric.startswith('lambda'): self.metric = eval(metric)
            else: self.metric = get_class_from_module_path(metric)
        #end if
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, validation_data=None, **kwargs):
        klass = get_class_from_module_path(self.classifier)

        if validation_data is None:  # use 0.2 for validation data
            X_train, X_validation, Y_train, Y_validation = train_test_split(X_featurized, Y_binarized, test_size=self.validation_size)
            logger.info('Using {} of training data ({} instances) for validation.'.format(self.validation_size, Y_validation.shape[0]))
        else:
            X_train, X_validation, Y_train, Y_validation = X_featurized, validation_data[0], Y_binarized, validation_data[1]
        #end if

        best_score, best_param = 0.0, None

        if self.n_jobs > 1: logger.info('Performing hyperparameter gridsearch in parallel using {} jobs.'.format(self.n_jobs))
        else: logger.debug('Performing hyperparameter gridsearch in parallel using {} jobs.'.format(self.n_jobs))

        param_scores = Parallel(n_jobs=self.n_jobs)(delayed(_fit_classifier)(klass, self.classifier_args, param, self.metric, X_train, Y_train, X_validation, Y_validation) for param in ParameterGrid(self.param_grid))

        best_param, best_score = max(param_scores, key=lambda x: x[1])
        logger.info('Best scoring param is {} with score {}.'.format(best_param, best_score))

        classifier_args = {}
        classifier_args.update(self.classifier_args)
        classifier_args.update(best_param)
        self.classifier_ = klass(**classifier_args)
        logger.info('Fitting final model <{}> on full data with param {}.'.format(self.classifier_, best_param))
        self.classifier_.fit(X_featurized, Y_binarized)

        return self
    #end def
#end class


def _fit_classifier(klass, classifier_args, param, metric, X_train, Y_train, X_validation, Y_validation):
    local_classifier_args = {}
    local_classifier_args.update(classifier_args)
    local_classifier_args.update(param)

    classifier = klass(**local_classifier_args).fit(X_train, Y_train)

    Y_predict = classifier.predict(X_validation)
    score = metric(Y_validation, Y_predict)
    logger.info('<{}> with param {} has micro F1 of {}.'.format(classifier, param, score))

    return (param, score)
#end def
