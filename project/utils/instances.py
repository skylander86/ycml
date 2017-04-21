from argparse import ArgumentParser
from collections import Counter
import json
import logging

from tabulate import tabulate

from .uriutils import URIFileType

__all__ = ['load_instances']

logger = logging.getLogger(__name__)


def load_instances(instance_files, labels_field=None, limit=None):
    count = 0
    for f in instance_files:
        for lineno, line in enumerate(f, start=1):
            o = json.loads(line)

            labels = None

            if labels_field is not None:
                labels = o[labels_field]
                del o[labels_field]
            #end if

            yield (o, labels)
            count += 1
            if count == limit: break
        #end for
        logger.info('{} lines read from instance file <{}>.'.format(lineno, f.name))

        if count == limit: break
    #end for
#end def


def main():
    parser = ArgumentParser(description='Quick utility for getting instance information.')
    parser.add_argument('instances', type=URIFileType(encoding='ascii'), nargs='+', metavar='<instances>', help='List of instance files to get information about.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    labels_freq = Counter()
    for f in A.instances:
        X, Y_labels = list(zip(*load_instances([f], labels_field='labels')))
        freq = Counter(label for labels in Y_labels for label in labels)
        freq['<none>'] = sum(1 for labels in Y_labels if not labels)

        logger.info('Label frequencies for <{}>:\n{}'.format(f.name, tabulate(freq.most_common() + [('Labels total', sum(freq.values())), ('Cases total', len(X))], headers=('Label', 'Freq'), tablefmt='psql')))
        labels_freq += freq
    #end for

    if len(A.instances) > 1:
        logger.info('Total label frequencies:\n{}'.format(tabulate(labels_freq.most_common() + [('Labels total', sum(labels_freq.values())), ('Cases total', len(X))], headers=('Label', 'Freq'), tablefmt='psql')))
#end def


if __name__ == '__main__': main()
