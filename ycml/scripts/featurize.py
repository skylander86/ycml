from argparse import ArgumentParser
from importlib import import_module
import logging
import sys

import numpy as np

from scipy.sparse import issparse

from tabulate import tabulate

from ..featurizers import load_featurizer, load_featurized, save_featurized
from ..utils import load_instances, URIFileType, load_dictionary_from_file, get_settings, parse_n_jobs

__all__ = []

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Featurize instances for ML classification.')
    parser.add_argument('featurizer_type', type=str, metavar='<featurizer_type>', nargs='?', default=None, help='Name of featurizer model to use.')
    parser.add_argument('-i', '--instances', type=URIFileType(encoding='ascii'), nargs='*', default=[], metavar='<instances>', help='List of instance files to featurize.')
    parser.add_argument('-o', '--output', type=URIFileType('wb'), metavar='<features_uri>', help='Save featurized instances here.')
    parser.add_argument('-s', '--settings', type=URIFileType('r'), metavar='<settings_uri>', help='Settings file to configure models.')
    parser.add_argument('--n-jobs', type=int, metavar='<N>', help='No. of processes to use during featurization.')
    parser.add_argument('--log-level', type=str, metavar='<log_level>', help='Set log level of logger.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle ordering of instances before writing them to file.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--fit', type=URIFileType('wb'), metavar='<featurizer>', help='Fit instances and save featurizer model file here.')
    group.add_argument('-t', '--featurize', type=URIFileType('rb'), metavar='<featurizer_uri>', help='Use this featurizer to transform instances.')
    group.add_argument('-z', '--featurizer-info', type=URIFileType('rb'), nargs='+', metavar='<featurizer_uri>', help='Display information about featurizer model.')
    group.add_argument('-x', '--features-info', type=URIFileType('rb'), nargs='+', metavar='<featurized_uri>', help='Display information about featurized instance file.')
    group.add_argument('-v', '--verify', type=URIFileType('rb'), metavar=('<featurizer_uri>', '<featurized_uri>'), nargs=2, help='Verify that the featurized instance file came from the same featurizer model.')
    A = parser.parse_args()

    file_settings = load_dictionary_from_file(A.settings) if A.settings else {}

    log_level = get_settings(key='log_level', sources=(A, 'env', file_settings), default='DEBUG').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    featurizer_type = get_settings(key='featurizer_type', sources=(A, 'env', file_settings))
    featurizer_parameters = get_settings(key='featurizer_parameters', sources=(file_settings, ), default={})
    featurizer_parameters['n_jobs'] = parse_n_jobs(get_settings(key='n_jobs', sources=(A, 'env', featurizer_parameters, file_settings), default=1))
    labels_field = get_settings(key='labels_field', sources=('env', file_settings), default='labels')
    logger.debug('Using "{}" for labels field.'.format(labels_field))

    if A.instances:
        X, Y_labels = list(map(np.array, zip(*load_instances(A.instances, labels_field=labels_field))))

    if A.fit:
        if not featurizer_type: parser.error('featurizer_type needs to be specified for fitting.'.format(featurizer_type))
        module_path, class_name = featurizer_type.rsplit('.', 1)
        module = import_module(module_path)
        model_class = getattr(module, class_name)
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
            info_table = [
                ('UUID', featurizer_uuid),
                ('Featurized at', str(featurized_at) + ' UTC'),
            ]
            if issparse(X_featurized):
                info_table += [
                    ('Matrix type', 'Sparse'),
                    ('Matrix dimensions', '{}x{} ({:,} active features)'.format(X_featurized.shape[0], X_featurized.shape[1], len(X_featurized.data))),
                    ('Matrix sparsity', '{:.3f}%'.format(len(X_featurized.data) / (X_featurized.shape[0] * X_featurized.shape[1]) * 100.0)),
                ]
            else:
                info_table += [('Matrix type', 'Dense'), ('Matrix dimensions', '{}x{}'.format(X_featurized.shape[0], X_featurized.shape[1]))]
            #end if

            logger.info('Featurizer info for <{}>:\n{}'.format(f.name, tabulate(info_table, headers=('Key', 'Value'), tablefmt='psql')))
        #end for

    elif A.verify:
        featurizer = load_featurizer(A.verify[0])
        featurizer_uuid = load_featurized(A.verify[1], ('featurizer_uuid',))[0]
        if featurizer_uuid == featurizer.uuid: logger.info('UUID match OK.')
        else:
            logger.error('UUID mismatch! Featurizer UUID {} != {}'.format(featurizer.uuid, featurizer_uuid))
            sys.exit(-1)
    #end if

    if A.fit or A.featurize:
        logger.debug('Feature matrix has dimensions {}x{} ({:,} active features).'.format(X_featurized.shape[0], X_featurized.shape[1], len(X_featurized.data)))

        if A.output:
            if A.shuffle:
                shuffled_indexes = np.random.permutation(X.shape[0])
                logger.info('Featurized instances shuffled.')
            else:
                shuffled_indexes = np.arange(X.shape[0])
            #end if

            id_key = None
            for key in ['_id', 'id', 'uuid']:
                if key in X[0]:
                    id_key = key
                    break
                #end if
            #end for
            if not id_key: raise logger.warning('Unable to find ID key in instances.')
            else: logger.info('Using "{}" as key for ID field.'.format(id_key))

            X_meta = np.array([dict(id=X[i][id_key]) for i in shuffled_indexes], dtype=np.object)
            save_featurized(A.output, X_featurized=X_featurized[shuffled_indexes, :], Y_labels=Y_labels[shuffled_indexes], X_meta=X_meta, featurizer_uuid=featurizer.uuid_)
        #end if
    #end if
#end def


if __name__ == '__main__': main()
