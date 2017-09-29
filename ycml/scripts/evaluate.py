from argparse import ArgumentParser
import logging
import re

import numpy as np

from sklearn.metrics import classification_report as sklearn_classification_report

from tabulate import tabulate

from uriutils import URIFileType, URIType

from ycsettings import Settings

from ..classifiers import load_classifier, get_thresholds_from_file
from ..featurizers import load_featurized
from ..utils import classification_report, find_best_thresholds, generate_pr_curves
from ..utils import save_dictionary_to_file

__all__ = []

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Evaluate and analyze the performance of a trained classifier.')
    parser.add_argument('-s', '--settings', type=URIFileType(), metavar='<settings_file>', help='Settings file to configure models.')

    parser.add_argument('--classifier-info', type=URIFileType(), metavar='<classifier_file>', help='Display information about classifier.')

    parser.add_argument('-c', '--classifier', type=URIFileType(), metavar='<classifier_file>', help='Classifier file to use for evaluation.')
    parser.add_argument('-f', '--featurized', type=URIFileType(), metavar='<featurized_file>', nargs='+', help='Featurized instances for evaluation.')
    parser.add_argument('-t', '--thresholds', type=URIFileType(), metavar='<thresholds>', help='Threshold file to use for prediction.')

    parser.add_argument('--load-probabilities', type=URIFileType('rb'), metavar='<probabilities_file>', help='Load probabilities from here instead of recalculating.')
    parser.add_argument('--save-probabilities', type=URIFileType('wb'), metavar='<probabilities_file>', help='Save evaluation probabilities; useful for calibration.')

    parser.add_argument('--min-precision', type=float, metavar='<precision>', default=None, help='Set the minimum precision threshold.')
    parser.add_argument('--best-thresholds', type=URIFileType('w'), metavar='<thresholds_file>', help='Save best F1 threshold values here.')
    parser.add_argument('--minprec-thresholds', type=URIFileType('w'), metavar='<thresholds_file>', help='Save minimum precision best F1 threshold values here.')
    parser.add_argument('--clip-thresholds', type=float, nargs=2, metavar=('lower', 'upper'), default=None, help='Clip threshold values to given range. Only applied when saving to file.')
    parser.add_argument('--pr-curves', type=URIType(), metavar='<folder>', help='Save precision-recall curves in this folder.')

    A = parser.parse_args()

    settings = Settings(A)

    log_level = settings.get('log_level', default='DEBUG').upper()
    log_format = settings.get('log_format', default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    if A.classifier_info:
        classifier = load_classifier(A.classifier_info)
        params = classifier.get_params(deep=True)

        if logger.getEffectiveLevel() > logging.INFO: logger.error('Log level should be at least INFO to view classifier info.')

        tabulate_rows = [
            ('Type', type(classifier).__name__),
            ('UUID', classifier.uuid_),
            ('Fitted at', classifier.fitted_at_),
            ('Output labels', ', '.join(classifier.classes_)),
        ]
        tabulate_rows += [('Model Parameters', '{} = {}'.format(k, v)) for k, v in sorted(params.items())]
        tabulate_rows += [('Fitted Parameters', '{} = {}'.format(k, getattr(classifier, k))) for k in sorted(dir(classifier)) if re.match(r'.+[a-z]_$', k, flags=re.I) and k not in ['fitted_at_', 'uuid_', 'classes_', 'input_dims_', 'output_dims_']]
        logger.info('Classifier info for <{}>:\n{}'.format(A.classifier_file.name, tabulate(tabulate_rows, headers=('Key', 'Value'), tablefmt='psql')))
    #end if

    thresholds = None

    if A.load_probabilities:
        o = np.load(A.load_probabilities)
        logger.info('Loaded probabalities from <{}>.'.format(A.load_probabilities.name))

        Y_proba = o['Y_proba']
        Y_true_binarized = o['Y_true_binarized']
        thresholds = o['thresholds']
        if thresholds.dtype == np.object: thresholds = None
        labels = o['labels']

        if A.thresholds: thresholds = get_thresholds_from_file(A.thresholds, labels)
        classifier = 'Classifier({})'.format(o['classifier_uuid'])

    else:
        classifier = load_classifier(A.classifier)
        labels = classifier.classes_
        if A.thresholds: thresholds = get_thresholds_from_file(A.thresholds, labels)

        featurizer_uuid = None
        X_featurized, Y_labels = None, None
        for f in A.featurized:
            X, Y, uuid = load_featurized(f, keys=('X_featurized', 'Y_labels', 'featurizer_uuid'))
            if featurizer_uuid is None:
                featurizer_uuid = uuid
                X_featurized = X
                Y_labels = Y
            else:
                if uuid != featurizer_uuid: raise ValueError('<{}> has featurizer UUID "{}" which is inconsistent with <{}> UUID "{}".'.format(f.name, uuid, A.featurized[0].name, featurizer_uuid))
                X_featurized = np.concatenate((X_featurized, X), axis=0)
                Y_labels = np.concatenate((Y_labels, Y), axis=0)
            #end if
        #end for
        # X_featurized, Y_labels, featurizer_uuid = load_featurized(A.featurized, keys=('X_featurized', 'Y_labels', 'featurizer_uuid'))
        Y_proba, _ = classifier.predict_and_proba(X_featurized, thresholds=thresholds, binarized=True)
        Y_true_binarized = classifier.binarize_labels(Y_labels)

        N = X_featurized.shape[0]
        assert Y_true_binarized.shape[0] == N
    #end if

    logger.info('Classification report for <{}>:\n{}'.format(
        str(classifier),
        classification_report(Y_true_binarized, Y_proba, target_names=labels, thresholds=thresholds, precision_thresholds=A.min_precision, order='support'))
    )

    # logger.info('sklearn Classification report for <{}>:\n{}'.format(
    #     str(classifier),
    #     sklearn_classification_report(Y_true_binarized, Y_proba, target_names=labels, digits=3))
    # )

    if A.save_probabilities:
        np.savez_compressed(A.save_probabilities, featurizer_uuid=featurizer_uuid, classifier_uuid=classifier.uuid_, Y_proba=Y_proba, Y_true_binarized=Y_true_binarized, thresholds=thresholds, labels=labels)
        logger.info('Saved evaluation probabilities to <{}>.'.format(A.save_probabilities.name))
    #end if

    if A.best_thresholds:
        best_thresholds = find_best_thresholds(Y_true_binarized, Y_proba, target_names=labels)
        if A.clip_thresholds:
            best_thresholds = best_thresholds.clip(A.clip_thresholds[0], A.clip_thresholds[1])
            logger.info('Best threshold values clipped to [{}, {}].'.format(*A.clip_thresholds))
        #end if

        o = dict((c, float(best_thresholds[i])) for i, c in enumerate(labels))
        save_dictionary_to_file(A.best_thresholds, o, title='thresholds')
    #end if

    if A.minprec_thresholds:
        minprec_thresholds = find_best_thresholds(Y_true_binarized, Y_proba, precision_thresholds=A.min_precision, target_names=labels)
        if A.clip_thresholds:
            minprec_thresholds = minprec_thresholds.clip(A.clip_thresholds[0], A.clip_thresholds[1])
            logger.info('Min-precision threshold values clipped to [{}, {}].'.format(*A.clip_thresholds))
        #end if

        o = dict((c, float(minprec_thresholds[i])) for i, c in enumerate(labels))
        save_dictionary_to_file(A.minprec_thresholds, o, title='thresholds')
    #end if

    if A.pr_curves:
        generate_pr_curves(Y_true_binarized, Y_proba, A.pr_curves.geturl(), target_names=labels, thresholds=thresholds, precision_thresholds=A.min_precision)
    #end if
#end def


if __name__ == '__main__': main()
