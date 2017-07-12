__all__ = ['HelloWorldFeaturizer']

import numpy as np

from ycml.featurizers import BaseFeaturizer
from ycml.transformers import PureTransformer


class RandomFeatures(PureTransformer):
    def __init__(self, feature_dims=10, **kwargs):
        self.feature_dims = feature_dims

    def transform(self, X):
        return np.random.rand(len(X), self.feature_dims)
#end class


class HelloWorldFeaturizer(BaseFeaturizer):
    def __init__(self, feature_dims=10, **kwargs):
        super(HelloWorldFeaturizer, self).__init__(steps=[('random_features', RandomFeatures(feature_dims=feature_dims))], **kwargs)

        self.feature_dims = feature_dims
    #end def
#end class
