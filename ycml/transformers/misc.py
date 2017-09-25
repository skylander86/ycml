__all__ = ['DictExtractionTransformer', 'DictListExtractionTransformer', 'FunctionTransformer', 'IdentityTransformer']

from . import PureTransformer


class DictExtractionTransformer(PureTransformer):
    """Extract a given key from dictionary object."""

    def __init__(self, key=None, default=None, **kwargs):
        kwargs.setdefault('nparray', False)
        super(DictExtractionTransformer, self).__init__(**kwargs)

        self.key = key
        self.default = default
    #end def

    def transform_one(self, d, **kwargs): return d.get(self.key, self.default)
#end class


class DictListExtractionTransformer(PureTransformer):
    """Extract a given key from list of dictionary object."""

    def __init__(self, key=None, default=None, **kwargs):
        super(DictListExtractionTransformer, self).__init__(**kwargs)

        self.key = key
        self.default = default
    #end def

    def transform_one(self, L, **kwargs): return [d.get(self.key, self.default) for d in L]
#end class


class FunctionTransformer(PureTransformer):
    def __init__(self, func, **kwargs):
        super(FunctionTransformer, self).__init__(**kwargs)
        self.func = func
    #end def

    def transform_one(self, x, **kwargs):
        return self.func(x)
#end class


class IdentityTransformer(PureTransformer):
    def transform_one(self, x, **kwargs):
        return x
#end class
