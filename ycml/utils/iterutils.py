__all__ = ['chunked_iterator']

from itertools import islice


def chunked_iterator(iterable, n):
    it = iter(iterable)
    while True:
       chunk = tuple(islice(it, n))
       if not chunk: return
       yield chunk
    #end while
#end def
