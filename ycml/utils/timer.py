_all__ = ['Timer']

import time


class Timer(object):
    def __init__(self, display_format='(took {elapsed:.3f} {unit})', display_unit='seconds'):
        self.start_time = time.time()
        self.display_unit = display_unit
        self.display_format = display_format
    #end def

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        if self.display_unit == 'seconds': elapsed = self.elapsed_time
        elif self.display_unit == 'milliseconds': elapsed = self.elapsed_time * 1000.0

        return self.display_format.format(elapsed=elapsed, unit=self.display_unit)
    #end def
#end class
