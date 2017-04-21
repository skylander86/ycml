from argparse import ArgumentParser, FileType
import logging
import sys

import numpy as np

from tabulate import tabulate

from ..featurizers import load_featurizer, load_featurized, save_featurized
from ..utils import load_instances, autogz_type, load_dictionary_from_file, get_settings

__all__ = []

logger = logging.getLogger(__name__)

FEATURIZERS_MAP = {}


def main():
    parser = ArgumentParser(description='Featurize instances for ML classification.')
    parser.add_argument('featurizer_type', type=str, metavar='<featurizer_type>', nargs='?', choices=FEATURIZERS_MAP.keys(), help='Name of featurizer model to use.')
    parser.add_argument('-i', '--instances', type=autogz_type(), nargs='*', default=[], metavar='<instances>', help='List of instance files to featurize.')
    parser.add_argument('-o', '--output', type=FileType('wb'), metavar='<features_file>', help='Save featurized instances here.')
    parser.add_argument('-s', '--settings', type=FileType('r'), metavar='<settings_file>', help='Settings file to configure models.')
    parser.add_argument('--n-jobs', type=int, metavar='<N>', help='No. of processes to use during featurization.')
    parser.add_argument('--log-level', type=str, metavar='<log_level>', help='Set log level of logger.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--fit', type=autogz_type('w'), metavar='<featurizer>', help='Fit instances and save featurizer model file here.')
    group.add_argument('-t', '--featurize', type=autogz_type(), metavar='<featurizer>', help='Use this featurizer to transform instances.')
    group.add_argument('-z', '--featurizer-info', type=autogz_type(), nargs='+', metavar='<featurizer>', help='Display information about featurizer model.')
    group.add_argument('-x', '--features-info', type=FileType('rb'), nargs='+', metavar='<features_file>', help='Display information about featurized instance file.')
    group.add_argument('-v', '--verify', type=autogz_type(), metavar=('<featurizer>', '<features_file>'), nargs=2, help='Verify that the featurized instance file came from the same featurizer model.')

    A = parser.parse_args()

    file_settings = load_dictionary_from_file(A.settings) if A.settings else {}

    log_level = get_settings(key='log_level', sources=(A, 'env', file_settings), default='DEBUG').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    featurizer_type = get_settings(key='featurizer_type', sources=(A, 'env', file_settings))
    featurizer_parameters = get_settings(key='featurizer_parameters', source=(file_settings, ), default={})
    featurizer_parameters['n_jobs'] = get_settings(key='n_jobs', sources=(A, 'env', featurizer_parameters, file_settings), default=1)

    if A.instances:
        X, Y_labels = list(zip(*load_instances(A.instances, labels_field='aols')))

    if A.fit:
        model_class = FEATURIZERS_MAP.get(featurizer_type)
        if not model_class: parser.error('Unknown featurizer model "{}".'.format(featurizer_type))

        featurizer = model_class(**featurizer_parameters)
        X_featurized = featurizer.fit_transform(X)
        featurizer.save(A.fit)

    elif A.featurize:
        featurizer = load_featurizer(A.featurize)
        if not A.output: logger.warning('No output file specified! The --featurize option is often used in conjunction with --output.')
        X_featurized = featurizer.transform(X)

    elif A.featurizer_info:
        if logger.getEffectiveLevel() > logging.INFO: logger.error('Log level should be at most INFO to view featurizer info.')
        for f in A.featurizer_info:
            featurizer = load_featurizer(f)
            logger.info('Featurizer info for <{}>:\n{}'.format(f.name, tabulate([
                ('Type', type(featurizer).__name__),
                ('UUID', featurizer.uuid),
            ], headers=('Key', 'Value'), tablefmt='psql')))
        #end for

    elif A.features_info:
        if logger.getEffectiveLevel() > logging.INFO: logger.error('Log level should be at most INFO to view featurized instances info.')
        for f in A.features_info:
            featurizer_uuid, X_featurized, featurized_at = load_featurized(f, ('featurizer_uuid', 'X_featurized', 'featurized_at'))
            logger.info('Featurizer info for <{}>:\n{}'.format(f.name, tabulate([
                ('UUID', featurizer_uuid),
                ('Featurized at', str(featurized_at) + ' UTC'),
                ('Matrix', '{}x{} ({:,} active features)'.format(X_featurized.shape[0], X_featurized.shape[1], len(X_featurized.data))),
                ('Matrix sparsity', '{:.3f}%'.format(len(X_featurized.data) / (X_featurized.shape[0] * X_featurized.shape[1]) * 100.0)),
            ], headers=('Key', 'Value'), tablefmt='psql')))
        #end for

    elif A.verify:
        featurizer = load_featurizer(A.verify[0])
        featurizer_uuid = load_featurized(A.verify[1], ('featurizer_uuid',))
        if featurizer_uuid == featurizer.uuid: logger.info('UUID match OK.')
        else:
            logger.error('UUID mismatch! Featurizer UUID {} != {}'.format(featurizer.uuid, featurizer_uuid))
            sys.exit(-1)
    #end if

    if A.fit or A.featurize:
        logger.debug('Feature matrix has dimensions {}x{} ({:,} active features).'.format(X_featurized.shape[0], X_featurized.shape[1], len(X_featurized.data)))

        if A.output:
            X_meta = np.array([dict(source=o['source'], id=o['id']) for i, o in enumerate(X)], dtype=np.object)
            save_featurized(A.output, featurizer.uuid, X_featurized=X_featurized, X_meta=X_meta, Y_labels=Y_labels)
        #end if
    #end if
#end def


if __name__ == '__main__': main()
