"""
This module provides wrapper function for transparently handling files regardless of location (local, cloud, etc).
"""

from contextlib import contextmanager
import gzip
from io import BytesIO, TextIOWrapper
import logging
import os
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

try: from urlparse import urlparse  # Python 2
except ImportError: from urllib.parse import urlparse  # Python 3

try:
    import boto3
    s3_client = boto3.client('s3')
except ImportError: boto3 = None

try: import requests
except ImportError: requests = None

__all__ = ['uri_open', 'uri_to_tempfile', 'uri_read', 'uri_dump', 'URIFileType', 'URIType']

logger = logging.getLogger(__name__)


def uri_open(uri, mode='rb', encoding='utf-8', use_gzip='auto', io_args={}, urifs_args={}):
    o = urlparse(uri)
    if o.scheme not in ['', 'file', 's3']: raise Exception('Unknown URI scheme {}'.format(o.scheme))
    elif o.scheme == 's3' and boto3 is None: raise Exception('S3 is not supported. You will need to install boto3.')
    elif o.scheme in ['http', 'https'] and requests is None: raise Exception('{} is not supported. You will need to install requests.'.format(o.scheme.upper()))

    _, ext = os.path.splitext(o.path)
    use_gzip = use_gzip == 'always' or use_gzip is True or (ext in ['.gz'] and use_gzip == 'auto')
    binary_mode = 'b' in mode
    read_mode = 'r' in mode

    if read_mode:
        if o.scheme == 's3':
            r = s3_client.get_object(Bucket=o.netloc, Key=o.path.lstrip('/'), **urifs_args)
            fileobj = BytesIO(r['Body'].read())  # future: Add support for local temp file

        elif o.scheme in ['http', 'https']:
            r = requests.get(uri)
            fileobj = BytesIO(r.content)
            if not binary_mode and not use_gzip: encoding = r.encoding

        elif not o.scheme or o.scheme == 'file':
            fpath = os.path.join(o.netloc, o.path.lstrip('/')).rstrip('/') if o.netloc else o.path
            fileobj = open(fpath, 'rb')
        #end if
    else:  # write mode
        if o.scheme == 's3':
            if use_gzip: urifs_args['ContentEncoding'] = 'gzip'
            fileobj = S3URIWriter(bucket=o.netloc, key=o.path.lstrip('/'), **urifs_args)

        elif o.scheme in ['http', 'https']:
            raise OSError('Write mode not supported for {}.'.format(o.scheme.upper()))

        elif not o.scheme or o.scheme == 'file':
            fpath = os.path.join(o.netloc, o.path.lstrip('/')).rstrip('/') if o.netloc else o.path
            fileobj = open(fpath, 'wb')
        #end if
    #end if

    if use_gzip: fileobj = gzip.GzipFile(fileobj=fileobj, mode='rb' if read_mode else 'wb')
    if not binary_mode: fileobj = TextIOWrapper(fileobj, encoding=encoding, **io_args)

    return fileobj
#end def


@contextmanager
def uri_to_tempfile(uri, *, delete=True, **kwargs):
    with uri_open(uri, mode='rb', use_gzip=False) as f_uri:
        with NamedTemporaryFile(mode='wb', prefix='uri.', delete=False) as f_temp:
            copyfileobj(f_uri, f_temp)
            f_temp_name = f_temp.name
        #end with
    #end with
    logger.debug('URI <{}> downloaded to temporary file <{}> ({} bytes).'.format(uri, f_temp_name, os.path.getsize(f_temp_name)))

    f = uri_open(f_temp_name, **kwargs)

    yield f

    f.close()
    if delete: os.remove(f_temp_name)
#end def


def uri_read(uri, mode='rb', encoding='utf-8', use_gzip='auto', io_args={}, urifs_args={}):
    with uri_open(uri, mode=mode, encoding=encoding, use_gzip=use_gzip, io_args=io_args, urifs_args=urifs_args) as f:
        content = f.read()
    return content
#end def


def uri_dump(uri, content, mode='wb', encoding='utf-8', use_gzip='auto', io_args={}, urifs_args={}):
    with uri_open(uri, mode=mode, encoding=encoding, use_gzip=use_gzip, io_args=io_args, urifs_args=urifs_args) as f:
        f.write(content)
#end def


class URIFileType(object):
    def __init__(self, mode='rb', **kwargs):
        self.kwargs = kwargs
        self.kwargs['mode'] = mode
    #end def

    def __call__(self, uri):
        return uri_open(uri, **self.kwargs)
#end class


class URIType(object):
    def __call__(self, uri):
        o = urlparse(uri)
        return o
    #end def
#end class


class S3URIWriter(BytesIO):
    def __init__(self, bucket, key, **urifs_args):
        super(S3URIWriter, self).__init__()
        self.bucket = bucket
        self.key = key
        self.urifs_args = urifs_args
    #end def

    @property
    def name(self): return 's3://{}/{}'.format(self.bucket, self.key)

    def close(self):
        s3_client.put_object(Bucket=self.bucket, Key=self.key, Body=self.getvalue(), **self.urifs_args)
        super(S3URIWriter, self).close()
    #end def
#end class
