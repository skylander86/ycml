"""
python -m helloworld.generate_data data/train.json.gz
python -m helloworld.generate_data data/evaluate.json.gz -N 200
"""

from argparse import ArgumentParser
from itertools import combinations
import json
import logging
from random import choice
from uuid import uuid4

from ycml.utils import URIFileType

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Generates a sample data set for demo purposes.')
    parser.add_argument('output', type=URIFileType('w'), metavar='<output>', help='Save generated dataset here.')
    parser.add_argument('-N', type=int, metavar='<N>', default=1000, help='No. of sample instances to generate.')
    parser.add_argument('--labels', type=str, nargs='+', default=['apple', 'banana', 'cherry'], metavar='<label>', help='No. of sample instances to generate.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    label_combis = sum((list(combinations(A.labels, r)) for r in range(len(A.labels) + 1)), [])

    for i in range(A.N):
        _id = str(uuid4())
        A.output.write(json.dumps(dict(content=_id, id=_id, labels=choice(label_combis)), sort_keys=True))
        A.output.write('\n')
    #end for
    logger.info('{} samples generated in <{}>.'.format(A.N, A.output.name))
#end def


if __name__ == '__main__': main()
