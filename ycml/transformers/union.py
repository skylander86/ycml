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
        if self.as_dict:
            transformed = []
            first = True
            # zip(*[step.transform(X, **kwargs) for k, step in self.steps.items()])
            for k, step in self.steps.items():
                step_transformed = step.transform(X, **kwargs)
                for i, t in enumerate(step_transformed):
                    if first:
                        transformed.append({k: t})
                    else:
                        transformed[i][k] = t
                #end for

                first = False
            #end for

        else:
            transformed = []
            first = True
            for k, step in self.steps.items():
                step_transformed = step.transform(X, y=y, **kwargs)
                for i, t in enumerate(step_transformed):
                    if first:
                        transformed.append([t])
                    else:
                        transformed[i].append(t)
                #end for
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
