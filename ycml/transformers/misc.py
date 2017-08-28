__all__ = ['DictExtractionTransformer', 'DictListExtractionTransformer', 'FunctionTransformer', 'ListConcatTransformer']

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


class ListConcatTransformer(PureTransformer):
    """Concatenate two lists for each instance."""

    def __init__(self, steps=[], **kwargs):
        kwargs.setdefault('nparray', False)
        super(ListConcatTransformer, self).__init__(**kwargs)

        self.steps = steps
    #end def

    def fit(self, *args, **kwargs):
        for step in self.steps:
            step.fit(*args, **kwargs)

        return self
    #end def

    def _transform(self, X, *args, **kwargs):
        N = len(X)
        transformed = [[] for i in range(N)]
        for step in self.steps:
            transformed_step = step.transform(X, *args, **kwargs)
            assert len(transformed_step) == N
            for j in range(N):
                transformed[j] += transformed_step[j]
        #end for

        return transformed
    #end def
#end class
