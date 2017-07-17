from argparse import ArgumentParser
import logging
import random

from uriutils import URIFileType

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Script to partition file by lines.')
    parser.add_argument('-i', '--instances', type=URIFileType('r'), nargs='+', metavar='<instances>', help='List of instance files to partition.')
    parser.add_argument('-s', '--sizes', type=float, nargs='+', required=True, default=[], metavar='<N>', help='Proportions/Number of instances in each partition.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle the instances before partitioning.')
    parser.add_argument('-p', '--prefix', type=str, default='partition', metavar='<prefix>', help='Prefix filename to use when saving partitions.')
    parser.add_argument('-o', '--output', type=URIFileType('w'), nargs='*', default=[], metavar='<output>', help='Save partitioned instances here.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    if A.prefix != 'partition' and A.output: parser.error('--prefix and --output cannot be used together.')
    if A.output: output_files = A.output
    else: output_files = [open(A.prefix + '{:02d}'.format(i), 'w') for i, s in enumerate(A.sizes)]
    if len(A.sizes) != len(output_files): parser.error('Number of arguments in --sizes and --output mismatch.')

    instances = []
    for f in A.instances:
        lines = [line.strip() for line in f]
        instances += lines
        logger.info('{} lines read from instance file <{}>.'.format(len(lines), f.name))
    #end for
    N = len(instances)
    logger.info('{} instances found.'.format(N))

    if A.shuffle:
        random.shuffle(instances)
        logger.info('Instances shuffled.')
    #end if

    Z = sum(A.sizes)
    N_partitions = len(A.sizes)

    output_lines = [int(s / Z * N) for s in A.sizes]
    leftovers = N - sum(output_lines)
    for i in range(leftovers):
        output_lines[i % N_partitions] += 1
    assert sum(output_lines) == N

    cur = 0
    for num_instances, g in zip(output_lines, output_files):
        for i in range(num_instances):
            g.write(instances[cur])
            g.write('\n')
            cur += 1
        #end for
        g.close()

        logger.info('Wrote {} instances to <{}>.'.format(num_instances, g.name))
    #end for
#end def


if __name__ == '__main__': main()
