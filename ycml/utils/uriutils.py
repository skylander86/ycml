"""
This module provides wrapper function for transparently handling files regardless of location (local, cloud, etc).
"""
__all__ = ['uri_open', 'uri_to_tempfile', 'uri_read', 'uri_dump', 'uri_exists', 'get_uri_metadata', 'uri_exists_wait', 'URIFileType', 'URIType']

import functools
import warnings

import uriutils

warnings.simplefilter('default', category=DeprecationWarning)


def _deprecated(new_func):
    @functools.wraps(new_func)
    def f(*args, **kwargs):
        warnings.warn('The use of uriutils module in ycml.utils is deprecated. Please use the uriutils module instead.', category=DeprecationWarning, stacklevel=1)

        return new_func(*args, **kwargs)
    #end def

    return f
#end def


uri_open = _deprecated(uriutils.uri_open)
uri_read = _deprecated(uriutils.uri_read)
uri_dump = _deprecated(uriutils.uri_dump)
uri_exists = _deprecated(uriutils.uri_exists)
get_uri_metadata = _deprecated(uriutils.get_uri_metadata)
uri_exists_wait = _deprecated(uriutils.uri_exists_wait)


def uri_to_tempfile(uri, *, delete=True, **kwargs):
    warnings.warn('The use of uriutils methods in ycml.utils is deprecated. Please use the uriutils module instead.', category=DeprecationWarning, stacklevel=3)
    return uri_open(uri, in_memory=False, delete_tempfile=delete, **kwargs)
#end def


class URIFileType(uriutils.URIFileType):
    def __init__(self, *args, **kwargs):
        warnings.warn('The use of uriutils methods in ycml.utils is deprecated. Please use the uriutils module instead.', category=DeprecationWarning, stacklevel=1)
        super(URIFileType, self).__init__(*args, **kwargs)
    #end def
#end class


class URIType(uriutils.URIType):
    def __init__(self, *args, **kwargs):
        warnings.warn('The use of uriutils methods in ycml.utils is deprecated. Please use the uriutils module instead.', category=DeprecationWarning, stacklevel=1)
        super(URIType, self).__init__(*args, **kwargs)
    #end def
#end class
