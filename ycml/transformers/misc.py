__all__ = ['DictExtractionTransformer']

from . import PureTransformer


class DictExtractionTransformer(PureTransformer):
    """Extract a given key from dictionary object."""

    def __init__(self, key=None, default=None, **kwargs):
        super(DictExtractionTransformer, self).__init__(**kwargs)
        self.set_params(key=key, default=default)
    #end def

    def get_params(self, deep=True):
        params = super(DictExtractionTransformer, self).get_params(deep=deep)
        params['key'] = getattr(self, 'key', None)
        params['default'] = getattr(self, 'default', None)

        return params
    #end def

    def set_params(self, **kwargs):
        super(DictExtractionTransformer, self).set_params(**kwargs)

        self.key = kwargs.pop('key', None)
        self.default = kwargs.pop('default', None)

        return self
    #end def

    def transform_one(self, d): return d.get(self.key, self.default)
#end class
