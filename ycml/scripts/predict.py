from argparse import ArgumentParser
import json
import logging
import sys

from uriutils import URIFileType

from ..utils import load_dictionary_from_file, load_instances, get_settings
from ..featclass import load_featclass

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

    if A.featclass: featclass = load_featclass(settings=load_dictionary_from_file(A.featclass))
    else: featclass = load_featclass(settings=file_settings, uri=get_settings(key='featclass_uri', sources=('env', file_settings)))

    for count, args in enumerate(featclass.predictions_generator(load_instances(A.instances, labels_field=None), include_proba=A.probabilities, unbinarized=True), start=1):
        if A.probabilities:
            o, Y_predict_list, Y_proba_dict = args
            o['predictions'] = Y_predict_list
            o['probabilities'] = Y_proba_dict
        else:
            o, Y_predict_list = args
            o['predictions'] = Y_predict_list
        #end if

        A.output.write(json.dumps(o))
        A.output.write('\n')

        if count % 100000 == 0: logger.info('Saved {} predictions.'.format(count))
    #end for

    logger.info('Saved {} predictions to <{}>.'.format(count, A.output.name))
#end def


if __name__ == '__main__': main()
