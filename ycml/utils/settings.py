import json
import logging
import os
import yaml

__all__ = ['load_dictionary_from_file', 'save_dictionary_to_file', 'get_settings']

logger = logging.getLogger(__name__)


def load_dictionary_from_file(f, title='dictionary'):
    if isinstance(f, str): f = open(f, 'r')

    _, ext = os.path.splitext(f.name.lower())
    if ext == '.json': o = json.load(f)
    elif ext == '.yaml': o = yaml.load(f)
    else: raise ValueError('<{}> is an unrecognized format for dictionary. Only JSON and YAML are supported right now.')

    logger.info('Loaded {} {} from <{}>.'.format(ext[1:].upper(), title, f.name))

    return o
#end def


def save_dictionary_to_file(f, d, title='dictionary', **kwargs):
    if isinstance(f, str): f = open(f, 'w')

    _, ext = os.path.splitext(f.name.lower())
    if ext == '.json': json.dump(d, f, indent=4, sort_keys=True)
    elif ext == '.yaml': yaml.dump(d, f, default_flow_style=False)
    else: raise ValueError('<{}> is an unrecognized format for dictionary. Only JSON and YAML are supported right now.')

    logger.info('Saved {} {} to <{}>.'.format(ext[1:].upper(), title, f.name))
#end def


ENV_SOURCE_KEYS = ['env', 'environment', 'ENV']


def get_settings(*source_key_pairs, **kwargs):
    default = kwargs.pop('default', None)
    raise_on_missing = kwargs.pop('raise_on_missing', None)
    parse_string_func = kwargs.pop('parse_string_func', None)

    key = kwargs.pop('key', None)
    sources = kwargs.pop('sources', None)

    if key and sources:
        source_key_pairs = list(source_key_pairs) + [(src, key) for src in sources]

    for src, key in source_key_pairs:
        v = None
        for key_ in (key, key.upper() if not key.isupper() else None, key.lower() if not key.islower() else None):
            if key_ is None: continue

            if hasattr(src, key_): v = getattr(src, key_)
            elif hasattr(src, 'get'): v = src.get(key_)
            elif src in ENV_SOURCE_KEYS: v = os.environ.get(key_)

            if v is not None:
                if isinstance(v, str) and parse_string_func: return parse_string_func(v)
                return v
            #end if
        #end for
    #end for

    if raise_on_missing:
        raise ValueError('Unable to find setting [{}].'.format(', '.join(sorted(set(key for _, key in source_key_pairs)))))

    return default
#end def
