from argparse import ArgumentParser
import csv
from io import TextIOWrapper
import json
import logging
import os
import sys

try: import yaml
except ImportError: pass

from ..classifiers import load_classifier, get_thresholds_from_file
from ..featurizers import load_featurized, save_featurized
from ..utils import get_settings, URIFileType
from ..utils import load_dictionary_from_file

__all__ = []

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Classify instances using ML classifier.')
    parser.add_argument('--log-level', type=str, metavar='<log_level>', help='Set log level of logger.')
    parser.add_argument('-s', '--settings', type=URIFileType(), metavar='<settings_file>', help='Settings file to configure models.')
    parser.add_argument('--n-jobs', type=int, metavar='<N>', help='No. of processor cores to use.')

    parser.add_argument('classifier_file', type=URIFileType(), metavar='<classifier_file>', help='Model file to use for prediction.')
    parser.add_argument('featurized_file', type=URIFileType(), metavar='<featurized_file>', help='Predict labels of featurized instances.')
    parser.add_argument('-t', '--thresholds', type=URIFileType('r'), metavar='<thresholds>', help='Threshold file to use for prediction.')
    parser.add_argument('-o', '--output', type=URIFileType('wb'), metavar='<prediction_file>', default=sys.stdout.buffer, help='Save results of prediction here.')
    parser.add_argument('--format', type=str, metavar='<format>', choices=('json', 'csv', 'tsv', 'txt', 'npz', 'yaml'), help='Prediction file format (defaults to file extension).')
    parser.add_argument('-p', '--probs', action='store_true', help='Also save prediction probabilities.')

    A = parser.parse_args()

    file_settings = load_dictionary_from_file(A.settings) if A.settings else {}

    log_level = get_settings(key='log_level', sources=(A, 'env', file_settings), default='DEBUG').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    output_file, output_format = A.output, None
    if A.format is not None: output_format = A.format
    else:
        if output_file.name.endswith('.gz'): _, output_format = os.path.splitext(output_file.name[:-3])
        else: _, output_format = os.path.splitext(output_file.name)

        if output_format.startswith('.'): output_format = output_format[1:].lower()
        else: output_format = 'json'
    #end if

    classifier = load_classifier(A.classifier_file)
    thresholds = get_thresholds_from_file(A.thresholds, classifier.classes_) if A.thresholds else None

    if output_format in ['csv', 'tsv', 'txt', '', 'json', 'yaml']:
        output_file = TextIOWrapper(output_file, encoding='utf-8')

    if output_format in ['csv', 'tsv', 'txt', '']:
        if A.probs:
            writer = csv.DictWriter(output_file, fieldnames=['ID'] + list(classifier.classes_) + ['p({})'.format(c) for c in classifier.classes_], dialect='excel' if output_format == '.csv' else 'excel-tab')
        else:
            writer = csv.DictWriter(output_file, fieldnames=['ID'] + list(classifier.classes_), dialect='excel' if output_format == '.csv' else 'excel-tab')

        def _write_prediction(id_, pred, proba):
            o = dict((c, '0.0') for c in classifier.classes_)
            o['ID'] = id_
            for c in pred: o[c] = str(1.0)
            if proba:
                for c, p in proba.items():
                    o['p({})'.format(c)] = str(p)
            #end if

            writer.writerow(o)
        #end def

    elif output_format in ['json']:
        def _write_prediction(id_, pred, proba):
            o = dict(id=id_, labels=pred)
            if proba: o['probabilities'] = proba
            output_file.write(json.dumps(o, ensure_ascii=True))
            output_file.write('\n')
        #end def

    elif output_format in ['yaml']:
        def _write_prediction(id_, pred, proba):
            o = dict(id=id_, labels=pred)
            if proba: o['probabilities'] = proba
            output_file.write('---\n')
            output_file.write(yaml.dump(o, default_flow_style=False))
        #end def

    elif output_format in ['npz']:
        def _write_prediction(id_, pred, proba): return

    else: parser.error('"{}" is an unknown output format.'.format(output_format))

    featurizer_uuid, X_featurized, X_meta = load_featurized(A.featurized_file, keys=['featurizer_uuid', 'X_featurized', 'X_meta'])
    Y_proba, Y_predict_binarized = classifier.predict_and_proba(X_featurized, thresholds=thresholds)
    Y_predict_labels = classifier.unbinarize_labels(Y_predict_binarized)

    for i in range(X_featurized.shape[0]):
        X_meta[i]['predicted'] = Y_predict_labels[i]
        if A.probs: X_meta[i]['probabilities'] = dict((c, float(Y_proba[i, j])) for j, c in enumerate(classifier.classes_))
        _write_prediction(X_meta[i]['id'], list(Y_predict_labels[i]), X_meta[i]['probabilities'] if A.probs else None)
    #end for

    if output_format in ['npz']: save_featurized(output_file, X_featurized, Y_labels=Y_predict_labels, featurizer_uuid=featurizer_uuid, X_meta=X_meta)

    logger.info('Saved {} predictions to <{}> in {} format.'.format(Y_predict_labels.shape[0], output_file.name, output_format.upper()))
#end def


if __name__ == '__main__': main()
