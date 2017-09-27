__all__ = ['load_dictionary_from_file', 'save_dictionary_to_file']

import json
import logging
import os
import pickle

try: import yaml
except ImportError: pass

from uriutils import uri_open

logger = logging.getLogger(__name__)


def load_dictionary_from_file(file_or_filename, *, force_format=None, title='dictionary'):
    if isinstance(file_or_filename, str): file_or_filename = uri_open(file_or_filename, 'rb')

    if force_format: ext = '.' + force_format
    else: _, ext = os.path.splitext(file_or_filename.name.lower())

    if ext == '.json': o = json.load(file_or_filename)
    elif ext == '.yaml': o = yaml.load(file_or_filename)
    elif ext == '.pickle': o = pickle.load(file_or_filename)
    else: raise ValueError('<{}> is an unrecognized format for dictionary. Only JSON and YAML are supported right now.')

    logger.info('Loaded {} {} from <{}>.'.format(ext[1:].upper(), title, file_or_filename.name))

    return o
#end def


def save_dictionary_to_file(f, d, *, force_format=None, title='dictionary', **kwargs):
    if isinstance(f, str): f = open(f, 'w')

    if force_format: ext = '.' + force_format
    else: _, ext = os.path.splitext(f.name.lower())

    if ext == '.json':
        kwargs.setdefault('indent', 4)
        kwargs.setdefault('sort_keys', True)
        json.dump(d, f, **kwargs)

    elif ext == '.yaml':
        kwargs.setdefault('default_flow_style', False)
        yaml.dump(d, f, **kwargs)

    elif ext == '.pickle':
        pickle.dump(d, f, **kwargs)

    else: raise ValueError('<{}> is an unrecognized format for {}. Only JSON and YAML are supported right now.'.format(title))

    logger.info('Saved {} {} to <{}>.'.format(ext[1:].upper(), title, f.name))
#end def
