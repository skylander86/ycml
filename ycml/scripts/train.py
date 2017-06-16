from argparse import ArgumentParser
from importlib import import_module
import logging

from ..featurizers import load_featurized
from ..utils import load_dictionary_from_file, get_settings, URIFileType

__all__ = []

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Classify instances using ML classifier.')
    parser.add_argument('--log-level', type=str, metavar='<log_level>', help='Set log level of logger.')
    parser.add_argument('-s', '--settings', type=URIFileType(), metavar='<settings_file>', help='Settings file to configure models.')
    parser.add_argument('--n-jobs', type=int, metavar='<N>', help='No. of processor cores to use.')

    parser.add_argument('classifier_type', type=str, metavar='<classifier_type>', nargs='?', help='Type of classifier model to fit.')
    parser.add_argument('-f', '--featurized', type=URIFileType(), metavar='<featurized>', help='Fit model on featurized instances.')
    parser.add_argument('-o', '--output', type=URIFileType('wb'), metavar='<classifier_file>', help='Save trained classifier model here.')

    parser.add_argument('-r', '--resume', type=str, metavar='<param>', nargs='+', help='Resume training from given file (takes multiple arguments depending on classifier).')
    parser.add_argument('-v', '--validation-data', type=URIFileType(), metavar='<featurized>', help='Use this as validation set instead of system defined one.')

    A = parser.parse_args()

    file_settings = load_dictionary_from_file(A.settings) if A.settings else {}

    log_level = get_settings(key='log_level', sources=(A, 'env', file_settings), default='DEBUG').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    classifier_type = get_settings(key='classifier_type', sources=(A, 'env', file_settings))
    classifier_parameters = get_settings((file_settings, 'classifier_parameters'), default={})
    classifier_parameters['n_jobs'] = get_settings(key='n_jobs', sources=(A, classifier_parameters, file_settings), default=1)

    try: module_path, class_name = classifier_type.rsplit('.', 1)
    except ValueError: parser.error('{} is not a valid classifier. You need to specify the full Python dotted path to the classifier class.')

    module = import_module(module_path)
    classifier_class = getattr(module, class_name)
    if not classifier_class: parser.error('Unknown classifier name "{}".'.format(classifier_type))

    X_featurized, Y_labels = load_featurized(A.featurized, keys=('X_featurized', 'Y_labels'))

    fit_args = {}
    if A.validation_data:
        X_validation, Y_validation = load_featurized(A.validation_data, keys=('X_featurized', 'Y_labels'))
        fit_args['validation_data'] = (X_validation, Y_validation)
    #end if

    if A.resume: fit_args['resume'] = A.resume

    classifier = classifier_class(**classifier_parameters).fit(X_featurized, Y_labels, fit_args=fit_args)

    if A.output: classifier.save(A.output)
#end def


if __name__ == '__main__': main()
