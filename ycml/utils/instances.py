__all__ = ['load_instances', 'shuffle_instances']

from argparse import ArgumentParser
from collections import Counter
import json
import logging

import numpy as np

from tabulate import tabulate

from uriutils import URIFileType

from .timer import Timer

logger = logging.getLogger(__name__)


def load_instances(instance_files, labels_field='labels', display_threshold=None, limit=None, progress_interval=None):
    if not isinstance(instance_files, (list, tuple)): instance_files = [instance_files]

    def _filter_display(d, threshold):
        if threshold is None:
            yield from d
        else:
            ignored_count, ignored_freq = 0, 0
            for label, freq in d:
                if freq < threshold:
                    ignored_count += 1
                    ignored_freq += freq
                else:
                    yield label, freq
                #end if
            #end for

            if ignored_count > 0:
                yield 'Ignored labels with freq < {}'.format(threshold), '{} ({} labels)'.format(ignored_freq, ignored_count)
        #end if
    #end def

    total_count = 0
    labels_freq = Counter()
    for f in instance_files:
        freq = Counter()
        count = 0
        timer = Timer()

        try:
            for lineno, line in enumerate(f, start=1):
                o = json.loads(line)

                if labels_field is None or labels_field is False:
                    yield o
                else:
                    labels = o.get(labels_field)
                    if labels is not None:
                        try:
                            if labels: freq.update(labels)
                            else: freq['<none>'] += 1
                        except TypeError: pass  # unhashable type
                    #end if

                    yield (o, labels)
                #end if

                count += 1
                total_count += 1
                if progress_interval and count % progress_interval == 0: logger.debug('{} instances read from <{}> so far.'.format(count, f.name))
                if limit and total_count == limit: break
            #end for
        except EOFError as e: logger.exception('Exception while reading instances. Will skip the remaining of <{}>.'.format(f.name))
        #end try

        logger.info('{} instances read from file <{}> {}.'.format(count, f.name, timer))

        if labels_field:
            labels_freq += freq
            threshold = freq.most_common()[99][1] if len(freq) > 100 and display_threshold is None else display_threshold
            logger.info('Label frequencies for <{}>:\n{}'.format(f.name, tabulate(list(_filter_display(freq.most_common(), threshold)) + [('Labels total', sum(freq.values())), ('Total', count)], headers=('Label', 'Freq'), tablefmt='psql')))
        #end if

        if limit and total_count >= limit: break
    #end for

    if labels_field and len(instance_files) > 1:
        threshold = freq.most_common()[99][1] if len(freq) > 100 and display_threshold is None else display_threshold
        logger.info('Total label frequencies:\n{}'.format(tabulate(list(_filter_display(labels_freq.most_common(), threshold)) + [('Labels total', sum(labels_freq.values())), ('Total', total_count)], headers=('Label', 'Freq'), tablefmt='psql')))
#end def


def shuffle_instances(X, Y, *, limit=None):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    if limit: indices = indices[:limit]

    return X[indices], Y[indices]
#end def


def main():
    parser = ArgumentParser(description='Quick utility for getting instance information.')
    parser.add_argument('instances', type=URIFileType(mode='r'), nargs='+', metavar='<instances>', help='List of instance files to get information about.')
    parser.add_argument('-l', '--label-key', type=str, metavar='<key>', default='labels', help='The key name for the label.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    list(load_instances(A.instances, labels_field=A.label_key))
#end def


if __name__ == '__main__': main()
