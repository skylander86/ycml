from __future__ import print_function, unicode_literals

from argparse import ArgumentParser
import logging
import os
import re
import time

from uriutils import URIFileType

logger = logging.getLogger(__name__)


VM_REGEX = re.compile(r'(Vm\w+)\:\s+(\d+) kB')
NAME_REGEX = re.compile(r'(Name)\:\s+(.+)$')
TRACKED_STATUS = ['VmPeak', 'VmRSS', 'VmSize']


def main():
    parser = ArgumentParser(description='Script to track memory usage of processes.')
    parser.add_argument('pid', type=int, metavar='<pid>', nargs='+', help='Process IDs to track.')
    parser.add_argument('--rate', type=float, metavar='<rate>', default=30, help='Sample every rate seconds.')
    parser.add_argument('--save', type=URIFileType('w'), metavar='<file>', help='Save sampled information to file.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s] %(levelname)s: %(message)s', level=logging.DEBUG)

    logger.info('Collecting memory information for PIDs {} at a rate of {} seconds.'.format(A.pid, A.rate))
    ignored_pid = set()
    if A.save: print('#', '\t'.join(['Timestamp', 'Pid-Name', 'Attribute', 'kB'], file=A.save))
    last_info = time.time()
    sample_count = 0
    have_data = True
    while True:
        timestamp = '{:.0f}'.format(time.time())
        for pid in A.pid:
            if pid in ignored_pid: continue
            status_file = os.path.join('/proc', str(pid), 'status')
            if not os.path.isfile(status_file):
                logger.warning('Process {} has disappeared. Will not track it anymore.'.format(pid))
                ignored_pid.add(pid)
                continue
            #end if

            status = {}
            with open(os.path.join('/proc', str(pid), 'status')) as f:
                for line in f:
                    m = NAME_REGEX.match(line.strip())
                    if m: status[m.group(1)] = m.group(2)
                    m = VM_REGEX.match(line)
                    if m: status[m.group(1)] = m.group(2)
                #end for
            #end with
            sample_count += 1

            proc_key = '{}-{}'.format(pid, status['Name'])
            s = ['Process={}'.format(proc_key)]
            for k in TRACKED_STATUS:
                if A.save:
                    print('\t'.join([timestamp, proc_key, k, status[k]]), file=A.save)
                s.append('{}={}'.format(k, convert_memory(status[k])))
            #end for
            logger.info('; '.join(s))
            have_data = True
        #end for

        if have_data and time.time() - last_info > A.rate * 10:
            logger.info('Collected {} samples so far...'.format(sample_count))
            last_info = time.time()
            have_data = False
        #end if

        time.sleep(A.rate)
    #end while
#end def


def convert_memory(kb_size):
    converted = float(kb_size)
    converted_unit = 'kB'

    for unit in ['MB', 'GB', 'TB', 'PB']:
        if converted > 1024:
            converted /= 1024.0
            converted_unit = unit
        else: break
    #end for

    return '{:.3f}{}'.format(converted, converted_unit)
#end def


if __name__ == '__main__': main()
