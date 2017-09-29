from argparse import ArgumentParser
import logging
import sys

import numpy as np

from scipy.sparse import issparse

from tabulate import tabulate

from uriutils import URIFileType

from ycsettings import Settings

from ..featurizers import load_featurizer, load_featurized, save_featurized
from ..utils import load_instances, get_class_from_module_path

__all__ = []

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Featurize instances for ML classification.')
    parser.add_argument('featurizer_type', type=str, metavar='<featurizer_type>', nargs='?', default=None, help='Name of featurizer model to use.')
    parser.add_argument('-i', '--instances', type=URIFileType('r'), nargs='*', default=[], metavar='<instances>', help='List of instance files to featurize.')
    parser.add_argument('-o', '--output', type=URIFileType('wb'), metavar='<features_uri>', help='Save featurized instances here.')
    parser.add_argument('-s', '--settings', dest='settings_uri', type=URIFileType(), metavar='<settings_uri>', help='Settings file to configure models.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle ordering of instances before writing them to file.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--fit', type=URIFileType('wb'), metavar='<featurizer>', help='Fit instances and save featurizer model file here.')
    group.add_argument('-t', '--featurize', type=URIFileType(), metavar='<featurizer_uri>', help='Use this featurizer to transform instances.')
    group.add_argument('-z', '--featurizer-info', type=URIFileType(), nargs='+', metavar='<featurizer_uri>', help='Display information about featurizer model.')
    group.add_argument('-x', '--features-info', type=URIFileType(), nargs='+', metavar='<featurized_uri>', help='Display information about featurized instance file.')
    group.add_argument('-v', '--verify', type=URIFileType(), metavar=('<featurizer_uri>', '<featurized_uri>'), nargs=2, help='Verify that the featurized instance file came from the same featurizer model.')
    A = parser.parse_args()

    settings = Settings(A, search_first=['env', 'env_settings_uri'])

    log_level = settings.get('log_level', default='DEBUG').upper()
    log_format = settings.get('log_format', default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    featurizer_type = settings.get('featurizer_type')
    featurizer_parameters = settings.getdict('featurizer_parameters', default={})
    featurizer_parameters['n_jobs'] = settings.getnjobs('n_jobs', default=1)

    labels_field = settings.get('labels_field', default='labels')
    logger.debug('Using "{}" for labels field.'.format(labels_field))

    if A.instances:
        X, Y_labels = list(map(np.array, zip(*load_instances(A.instances, labels_field=labels_field))))

    if A.fit:
        if not featurizer_type: parser.error('featurizer_type needs to be specified for fitting.'.format(featurizer_type))

        model_class = get_class_from_module_path(featurizer_type)
        if not model_class: parser.error('Unknown featurizer model "{}".'.format(featurizer_type))

        featurizer = model_class(**featurizer_parameters)
        X_featurized = featurizer.fit_transform(X, Y_labels)
        featurizer.save(A.fit)
        A.fit.close()

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
                ('Repr', repr(featurizer)),
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
            feature_count = X_featurized.shape[1] if X_featurized.ndim >= 2 else 1
            if issparse(X_featurized):
                info_table += [
                    ('Matrix type', 'Sparse'),
                    ('Matrix dimensions', '{}x{} ({:,} active features)'.format(X_featurized.shape[0], feature_count, len(X_featurized.data))),
                    ('Matrix sparsity', '{:.3f}%'.format(len(X_featurized.data) / (X_featurized.shape[0] * feature_count) * 100.0)),
                ]
            else:
                info_table += [('Matrix type', 'Dense'), ('Matrix dimensions', '{}x{}'.format(X_featurized.shape[0], feature_count))]
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
        logger.debug('Feature matrix has dimensions {} ({:,} active features).'.format('x'.join(map(str, X_featurized.shape)), len(X_featurized.data)))

        if A.output:
            if A.shuffle:
                shuffled_indexes = np.random.permutation(X.shape[0])
                logger.info('Featurized instances shuffled.')
            else:
                shuffled_indexes = np.arange(X.shape[0])
            #end if

            id_key = None
            for key in ['_id', 'id', 'id_', 'uuid', 'docid']:
                if key in X[0]:
                    id_key = key
                    break
                #end if
            #end for
            if not id_key: raise TypeError('Unable to find ID key in instances.')
            else: logger.info('Using "{}" as key for ID field.'.format(id_key))

            X_meta = np.array([dict(id=X[i][id_key]) for i in shuffled_indexes], dtype=np.object)
            save_featurized(A.output, X_featurized=X_featurized[shuffled_indexes, ...], Y_labels=Y_labels[shuffled_indexes], X_meta=X_meta, featurizer_uuid=featurizer.uuid_)
            A.output.close()
        #end if
    #end if
#end def


if __name__ == '__main__': main()
