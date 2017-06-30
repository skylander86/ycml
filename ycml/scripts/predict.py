from argparse import ArgumentParser
import json
import logging
import sys

import numpy as np

from ..utils import load_dictionary_from_file, load_instances, get_settings, URIFileType
from ..featclass import load_featclass

__all__ = []

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Predict instances using ML classifier.')
    parser.add_argument('-s', '--settings', type=URIFileType(), metavar='<settings_file>', help='Settings file to configure models.')
    parser.add_argument('instances', type=URIFileType('r'), metavar='<instances>', help='Instances to use for prediction.')
    parser.add_argument('--featclass', type=URIFileType(), metavar='<featclass_uri>', help='Featclass configuration file to use for prediction.')
    parser.add_argument('-p', '--probabilities', action='store_true', help='Also save prediction probabilities.')
    parser.add_argument('-o', '--output', type=URIFileType('w'), default=sys.stdout.buffer, help='Save predictions to this file.')

    A = parser.parse_args()

    file_settings = load_dictionary_from_file(A.settings) if A.settings else {}

    log_level = get_settings(key='log_level', sources=(A, 'env', file_settings), default='DEBUG').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    instances = list(load_instances([A.instances], labels_field=None))
    if A.featclass: featclass = load_featclass(settings=load_dictionary_from_file(A.featclass))
    else: featclass = load_featclass(settings=file_settings, uri=get_settings(key='featclass_uri', sources=('env', file_settings)))

    X = np.array(instances, dtype=np.object)
    logger.info('Predicting {} instances...'.format(X.shape[0]))
    if A.probabilities:
        Y_proba, Y_predict = featclass.predict_and_proba(X)
        Y_proba_dicts = featclass.classifier.unbinarize_labels(Y_proba, to_dict=True)
    else:
        Y_predict = featclass.predict(X)
        Y_proba_dicts = None
    #end if

    Y_predict_list = featclass.classifier.unbinarize_labels(Y_predict, to_dict=False)
    Y_predict, Y_proba = None, None

    for i in range(X.shape[0]):
        o = X[i]
        o['prediction'] = Y_predict_list[i]
        if Y_proba_dicts: o['probabilities'] = Y_proba_dicts[i]

        A.output.write(json.dumps(o))
        A.output.write('\n')

        if (i + 1) % 100000 == 0: logger.info('Saved {} predictions.'.format(i + 1))
    #end for

    logger.info('Saved {} predictions to <{}>.'.format(X.shape[0], A.output.name))
#end def


if __name__ == '__main__': main()
