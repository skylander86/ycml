from argparse import ArgumentParser
import csv
import json
import logging
import os
import re
import sys

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import classification_report

from tabulate import tabulate

from .classifiers import BinaryClassifier, load_classifier, get_thresholds_from_file
from .featurizers import load_featurized, save_featurized
from .utils import load_dictionary_from_file, get_settings, uri_open, URIFileType

__all__ = []

logger = logging.getLogger(__name__)


class ExampleClassifier(BinaryClassifier):
    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        self.classifier_ = SVC(probability=True).fit(X_featurized, Y_binarized)

        return self
    #end def

    def _predict_proba(self, X_featurized, **kwargs):
        return self.classifier_.predict_proba(X_featurized, **kwargs)
#end class


CLASSIFERS_MAP = {
    'ExampleClassifier': ExampleClassifier,
}


def main():
    parser = ArgumentParser(description='Classify instances using ML classifier.')
    parser.add_argument('--log-level', type=str, metavar='<log_level>', help='Set log level of logger.')
    parser.add_argument('-s', '--settings', type=URIFileType('r'), metavar='<settings_file>', help='Settings file to configure models.')
    parser.add_argument('-c', '--classifier-info', type=URIFileType('rb'), metavar='<classifier_file>', help='Display information about classifier.')
    parser.add_argument('--n-jobs', type=int, metavar='<N>', help='No. of processor cores to use.')

    subparsers = parser.add_subparsers(title='Different classifier modes for fitting, evaluating, and prediction.', metavar='<mode>', dest='mode')

    fit_parser = subparsers.add_parser('fit', help='Fit a classifier.')
    fit_parser.add_argument('classifier_type', type=str, metavar='<classifier_type>', help='Type of classifier model to fit.')
    fit_parser.add_argument('-f', '--featurized', type=URIFileType('rb'), metavar='<featurized>', help='Fit model on featurized instances.')
    fit_parser.add_argument('-o', '--output', type=URIFileType('wb'), metavar='<classifier_file>', required=True, help='Save trained classifier model here.')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a classifier.')
    evaluate_parser.add_argument('classifier_file', type=URIFileType('rb'), metavar='<classifier_file>', help='Model file to use for evaluation.')
    evaluate_parser.add_argument('featurized_file', type=URIFileType('rb'), metavar='<featurized_file>', help='Evaluate model on featurized instances.')
    evaluate_parser.add_argument('-t', '--thresholds', type=URIFileType('r'), metavar='<thresholds>', required=False, help='Threshold file to use for prediction.')
    evaluate_parser.add_argument('-p', '--save-probabilities', type=URIFileType('wb'), metavar='<probabilities_file>', required=False, help='Save evaluation probabilities; useful for calibration.')

    predict_parser = subparsers.add_parser('predict', help='Predict using a classifier.')
    predict_parser.add_argument('classifier_file', type=URIFileType('rb'), metavar='<classifier_file>', help='Model file to use for prediction.')
    predict_parser.add_argument('featurized_file', type=URIFileType('rb'), metavar='<featurized_file>', help='Predict labels of featurized instances.')
    predict_parser.add_argument('-t', '--thresholds', type=URIFileType('r'), metavar='<thresholds>', required=False, help='Threshold file to use for prediction.')
    predict_parser.add_argument('-o', '--output', type=URIFileType('wb'), metavar='<prediction_file>', help='Save results of prediction here.')
    predict_parser.add_argument('-f', '--format', type=str, metavar='<format>', default=None, choices=('json', 'csv', 'tsv', 'txt', 'npz'), help='Save results of prediction using this format (defaults to file extension).')
    predict_parser.add_argument('-p', '--probs', action='store_true', help='Also save prediction probabilities.')

    info_parser = subparsers.add_parser('info', help='Display information regarding classifier.')
    info_parser.add_argument('classifier_file', type=URIFileType('rb'), metavar='<classifier_file>', help='Model file to display information onabout.')

    A = parser.parse_args()

    file_settings = load_dictionary_from_file(A.settings) if A.settings else {}

    log_level = get_settings(key='log_level', sources=(A, 'env', file_settings), default='DEBUG').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    if A.mode == 'fit':
        classifier_type = get_settings(key='classifier_type', sources=(A, 'env', file_settings))
        classifier_parameters = get_settings((file_settings, 'classifier_parameters'), default={})
        classifier_parameters['n_jobs'] = get_settings(key='n_jobs', sources=(A, classifier_parameters, file_settings), default=1)

        classifier_class = CLASSIFERS_MAP.get(classifier_type)
        if not classifier_class: parser.error('Unknown model name "{}".'.format(classifier_type))

        X_featurized, Y_labels = load_featurized(A.featurized, keys=('X_featurized', 'Y_labels'))

        classifier = classifier_class(**classifier_parameters).fit(X_featurized, Y_labels, pos_labels='spam')

        classifier.save(A.output)

    elif A.mode == 'evaluate':
        classifier = load_classifier(A.classifier_file)

        thresholds = 0.5
        if A.thresholds:
            with uri_open(A.thresholds) as f:
                thresholds = get_thresholds_from_file(f, classifier.classes_)

        X_featurized, Y_labels, featurizer_uuid = load_featurized(A.featurized_file, keys=('X_featurized', 'Y_labels', 'featurizer_uuid'))
        Y_proba, Y_predict_binarized = classifier.predict_and_proba(X_featurized, thresholds=thresholds, binarized=True)
        Y_true_binarized = classifier.binarize_labels(Y_labels)

        N = X_featurized.shape[0]
        assert Y_predict_binarized.shape[0] == N
        assert Y_true_binarized.shape[0] == N

        # logger.info(
        #     'Evaluation results for <{}>:\n'.format(A.featurized_file.name) +
        #     evaluation_table(classifier.classes_, Y_true_binarized, Y_proba, thresholds)
        # )
        logger.info('Classification report:\n{}'.format(classification_report(Y_true_binarized, Y_predict_binarized, target_names=['not spam', 'spam'])))

        if A.save_probabilities:
            np.savez_compressed(A.save_probabilities, featurizer_uuid=featurizer_uuid, classifier_uuid=classifier.uuid_, Y_proba=Y_proba, Y_true_binarized=Y_true_binarized, thresholds=thresholds, labels=classifier.classes_)
            logger.info('Saved evaluation probabilities to <{}>.'.format(A.save_probabilities.name))
        #end if

    elif A.mode == 'predict':
        output_format = None
        if A.output is None:
            A.output = sys.stdout
            output_format = 'tsv'
        #end if

        if A.format is not None: output_format = A.format
        elif output_format is None:
            _, output_format = os.path.splitext(A.output.name)
            output_format = output_format[1:].lower()
        #end if

        classifier = load_classifier(A.classifier_file)

        if output_format in ['csv', 'tsv', 'txt', '']:
            if A.probs:
                writer = csv.DictWriter(A.output, fieldnames=['ID'] + list(classifier.classes_) + ['p({})'.format(c) for c in classifier.classes_], dialect='excel' if output_format == '.csv' else 'excel-tab')
            else:
                writer = csv.DictWriter(A.output, fieldnames=['ID'] + list(classifier.classes_), dialect='excel' if output_format == '.csv' else 'excel-tab')

            def _write_prediction(id_, pred, proba):
                o = dict((c, '0.0') for c in classifier.classes_)
                o['ID'] = id_
                for c in pred: o[c] = str(1.0)
                if proba:
                    for c, p in proba.items():
                        o['p({})'.format(c)] = str(p)
                #end if
                print(o)

                writer.writerow(o)
            #end def
        elif output_format in ['json']:
            def _write_prediction(id_, pred, proba):
                o = dict(id=id_, labels=pred)
                if proba: o['probabilities'] = proba
                A.output.write(json.dumps(o))
                A.output.write(b'\n')
            #end def
        elif output_format in ['npz']:
            def _write_prediction(id_, pred, proba):
                return

        else: parser.error('"{}" is an unknown output format.'.format(output_format))

        thresholds = 0.5

        featurizer_uuid, X_featurized, X_meta, Y_labels, featurized_at = load_featurized(A.featurized_file, keys=['featurizer_uuid', 'X_featurized', 'X_meta', 'Y_labels', 'featurized_at'])
        Y_proba, Y_predict_binarized = classifier.predict_and_proba(X_featurized, thresholds=thresholds)
        Y_predict_labels = classifier.unbinarize_labels(Y_predict_binarized)

        for i in range(X_featurized.shape[0]):
            X_meta[i]['predicted'] = Y_predict_labels[i]
            if A.probs: X_meta[i]['probabilities'] = dict((c, float(Y_proba[i, j])) for j, c in enumerate(classifier.classes_))
            _write_prediction(X_meta[i]['id'], Y_predict_labels[i], X_meta[i]['probabilities'] if A.probs else None)
        #end for

        if output_format in ['.npz']:
            save_featurized(A.output, featurizer_uuid, X_featurized, X_meta, Y_labels, featurized_at)

        logger.info('Saved predictions to <{}>.'.format(A.output.name))

    elif A.mode == 'info':
        classifier = load_classifier(A.classifier_file)
        params = classifier.get_params(deep=True)

        if logger.getEffectiveLevel() > logging.INFO: logger.error('Log level should be at least INFO to view classifier info.')

        tabulate_rows = [
            ('Type', type(classifier).__name__),
            ('UUID', classifier.uuid_),
            ('Fitted at', classifier.fitted_at_),
            ('Output labels', ', '.join(classifier.classes_)),
            ('Input x Output dims', '{} x {}'.format(classifier.input_dims_, classifier.output_dims_)),
        ]
        tabulate_rows += [('Model Parameters', '{} = {}'.format(k, v)) for k, v in sorted(params.items())]
        tabulate_rows += [('Fitted Parameters', '{} = {}'.format(k, getattr(classifier, k))) for k in sorted(dir(classifier)) if re.match(r'.+[a-z]_$', k, flags=re.I) and k not in ['fitted_at_', 'uuid_', 'classes_', 'input_dims_', 'output_dims_']]
        logger.info('Classifier info for <{}>:\n{}'.format(A.classifier_file.name, tabulate(tabulate_rows, headers=('Key', 'Value'), tablefmt='psql')))

    else: parser.error('Classifier mode must be one of {fit,evaluate,predict,info} is required.')
#end def


if __name__ == '__main__': main()
