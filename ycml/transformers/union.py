__all__ = ['ObjectFeatureUnion', 'ListConcatTransformer']

from collections import OrderedDict

import numpy as np

from . import PureTransformer


class ObjectFeatureUnion(PureTransformer):
    """Extract a given key from dictionary object."""

    def __init__(self, *args, as_dict=True, **kwargs):
        kwargs.setdefault('nparray', True)
        kwargs.setdefault('nparray_dtype', np.object)
        super(ObjectFeatureUnion, self).__init__(**kwargs)

        self.as_dict = as_dict
        self.steps = OrderedDict()

        for (k, step) in args:
            self.steps[k] = step
    #end def

    def fit(self, *args, **fit_params):
        for k, step in self.steps.items():
            step.fit(*args, **fit_params)

        return self
    #end def

    def _transform(self, X, y=None, **kwargs):
        N = len(X)
        if self.as_dict:
            transformed = [dict() for i in range(N)]
            for k, step in self.steps.items():
                step_transformed = step.transform(X, **kwargs)
                assert len(step_transformed) == N
                for i in range(N):
                    transformed[i][k] = step_transformed[i]
            #end for
        else:
            transformed = [list() for i in range(N)]
            for k, step in self.steps.items():
                step_transformed = step.transform(X, y=y, **kwargs)
                assert len(step_transformed) == N
                for i in range(N):
                    transformed[i].append(step_transformed[i])
            #end for
        #end if

        return transformed
    #end def
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
