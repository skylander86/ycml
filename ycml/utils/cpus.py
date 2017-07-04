__all__ = ['parse_n_jobs']

import logging
from multiprocessing import cpu_count
import re

logger = logging.getLogger(__name__)


def parse_n_jobs(s):
    n_jobs = None
    N = cpu_count()

    if isinstance(s, int): n_jobs = s

    elif isinstance(s, float): n_jobs = int(s)

    elif isinstance(s, str):
        m = re.match(r'(\d*(?:\.\d*)?)?(\s*\*?\s*n)?$', s.strip())
        if m is None: raise ValueError('Unable to parse n_jobs="{}"'.format(s))

        k = float(m.group(1)) if m.group(1) else 1
        if m.group(2): n_jobs = k * N
        elif k < 1: n_jobs = k * N
        else: n_jobs = int(k)

    else: raise TypeError('n_jobs argument must be of type str, int, or float.')

    n_jobs = int(n_jobs)
    if n_jobs <= 0:
        logger.warning('n_jobs={} is invalid. Setting n_jobs=1.'.format(n_jobs))
        n_jobs = 1
    #end if

    if n_jobs > N:
        logger.warning('n_jobs={} is > number of available CPUs ({}).'.format(s, N))

    return int(n_jobs)
#end def
