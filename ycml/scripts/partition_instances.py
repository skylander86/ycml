"""
This script is different from `ycml.scripts.partition_lines` because it takes labels into account and produces stratified partitions.
"""

from argparse import ArgumentParser
import json
import logging

from sklearn.model_selection import train_test_split

from ycml.utils import load_instances
from ycml.utils import URIFileType

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Script to partition instances in a stratified manner.')
    parser.add_argument('-i', '--instances', type=URIFileType('r'), nargs='+', metavar='<instances>', help='List of instance files to partition.')
    parser.add_argument('-l', '--label-key', type=str, metavar='<key>', default='labels', help='The key name for the label.')
    parser.add_argument('-s', '--train-size', type=float, required=True, default=0.7, metavar='<N>', help='Proportions of instances to use for training set.')
    parser.add_argument('-o', '--output', type=URIFileType('w'), nargs=2, required=True, metavar='<output>', help='Save partitioned instances here.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    X, Y_labels = load_instances(A.instances, labels_field=A.label_key)

    X_train, X_test = train_test_split(X, train_size=A.train_size, stratify=Y_labels)

    for o in X_train:
        A.output[0].write(json.dumps(o))
        A.output[0].write('\n')
    #end for
    logger.info('{} training instances written to <{}>.'.format(A.output[0].name))

    for o in X_test:
        A.output[1].write(json.dumps(o))
        A.output[1].write('\n')
    #end for
    logger.info('{} evaluation instances written to <{}>.'.format(A.output[1].name))
#end def


if __name__ == '__main__': main()
