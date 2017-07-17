__all__ = ['load_dictionary_from_file', 'save_dictionary_to_file', 'get_settings', 'bulk_get_settings']

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


ENV_SOURCE_KEYS = ['env', 'environment', 'ENV']


def get_settings(*source_key_pairs, key=None, keys=None, sources=None, default=None, raise_on_missing=None, parse_string_func=None, auto_parse=False):
    source_key_pairs = list(source_key_pairs)
    if key and sources:
        source_key_pairs += [(src, key) for src in sources]

    if keys and sources:
        source_key_pairs += [(src, k) for src in sources for k in keys]

    for src, key in source_key_pairs:
        v = None
        for key_ in (key, key.upper() if not key.isupper() else None, key.lower() if not key.islower() else None):
            if key_ is None: continue

            if hasattr(src, key_): v = getattr(src, key_)
            elif hasattr(src, 'get'): v = src.get(key_)
            elif src in ENV_SOURCE_KEYS: v = os.environ.get(key_)

            if v is not None:
                if isinstance(v, str):
                    if parse_string_func: return parse_string_func(v)
                    elif auto_parse: return _auto_parse(v)
                #end if

                return v
            #end if
        #end for
    #end for

    if raise_on_missing:
        raise ValueError('Unable to find setting [{}].'.format(', '.join(sorted(set(key for _, key in source_key_pairs)))))

    return default
#end def


def _auto_parse(s):
    try: return int(s)
    except ValueError: pass

    try: return float(s)
    except ValueError: pass

    try: return json.loads(s)
    except json.JSONDecodeError: pass

    return s
#end def


def bulk_get_settings(*sources, normalize_func=None, auto_parse=False):
    bulk_settings = {}
    chosen_sources = {}

    def _normalize_key(k):  # settings key are always lowercased. IRregardless.
        k = k.lower()
        normalized = normalize_func(k)
        return normalized if normalized else k
    #end def

    for i, src in enumerate(sources):
        if src in ENV_SOURCE_KEYS:
            for k, v in os.environ.items():
                norm_k = _normalize_key(k)
                bulk_settings[norm_k] = v
                chosen_sources[norm_k] = 'env'
            #end for
        elif hasattr(src, 'items'):
            for k, v in src.items():
                norm_k = _normalize_key(k)
                bulk_settings[norm_k] = v
                chosen_sources[norm_k] = 'dict_{}'.format(i)
            #end for

        else:
            for k in filter(lambda a: not a.startswith('_'), dir(src)):
                v = getattr(src, k)
                if callable(v): continue  # skip functions, we only want values

                norm_k = _normalize_key(k)
                bulk_settings[norm_k] = v
                chosen_sources[norm_k] = 'obj_{}'.format(i)
            #end for
        #end if
    #end for

    if auto_parse:
        for k, v in bulk_settings.items():
            if isinstance(v, str):
                bulk_settings[k] = _auto_parse(v)
    #end if

    return bulk_settings, chosen_sources
#end def
