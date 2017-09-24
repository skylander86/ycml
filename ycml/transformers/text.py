__all__ = ['CounterHashingVectorizer', 'ListHashingVectorizer', 'ListCountVectorizer', 'ListNGramAnalyzer', 'ListTfidfVectorizer', 'SpaceTokenizerTransformer', '_list_analyzer']

import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import PureTransformer

logger = logging.getLogger(__name__)


def _list_analyzer(L):
    for elem in L:
        yield elem
#end def


class ListNGramAnalyzer(PureTransformer):
    def __init__(self, ngram_range=(1, 1), ngram_delimiter=' ', **kwargs):
        kwargs.setdefault('nparray', False)
        super(ListNGramAnalyzer, self).__init__(**kwargs)

        if isinstance(ngram_range, tuple) and len(ngram_range) == 2: ngrams = list(range(ngram_range[0], ngram_range[1] + 1))
        elif isinstance(ngram_range, str): ngrams = [int(ngram_range)]
        elif isinstance(ngram_range, int): ngrams = [ngram_range]
        else: ngrams = ngram_range
        ngrams.sort()

        self.ngrams = ngrams
        self.ngram_delimiter = ngram_delimiter
    #end def

    def __call__(self, L):
        L = list(L)
        length = len(L)
        ngram_delimiter = self.ngram_delimiter

        for i in range(length):
            for n in self.ngrams:
                if i + n > length: break

                yield ngram_delimiter.join(L[i:i + n])
            #end for
        #end for
    #end def

    def transform_one(self, tokens, **kwargs):
        yield from self.__call__(tokens)
#end def


def _counter_analyzer(C):
    for k, v in C.items():
        for _ in range(v):
            yield k
#end def


class ListCountVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', ListNGramAnalyzer(**kwargs))
        super(ListCountVectorizer, self).__init__(**kwargs)
    #end def

    def fit(self, *args, **kwargs):
        ret = super(ListCountVectorizer, self).fit(*args)
        logger.debug('There are {} vocabulary items in <{}>.'.format(len(self.vocabulary_), self))

        return ret
    #end def

    def fit_transform(self, *args, **kwargs):
        transformed = super(ListCountVectorizer, self).fit_transform(*args)
        logger.debug('There {} vocabulary items in <{}>.'.format(len(self.vocabulary_), self))

        return transformed
    #end def
#end class


class ListTfidfVectorizer(TfidfVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', ListNGramAnalyzer(**kwargs))
        super(ListTfidfVectorizer, self).__init__(**kwargs)
    #end def

    def fit(self, *args, **kwargs):
        ret = super(ListTfidfVectorizer, self).fit(*args)
        logger.debug('There are {} vocabulary items in <{}>.'.format(len(self.vocabulary_), self))

        return ret
    #end def

    def fit_transform(self, X, *args, **kwargs):
        transformed = super(ListTfidfVectorizer, self).fit_transform(X)
        logger.debug('There {} vocabulary items in <{}>.'.format(len(self.vocabulary_), self))

        return transformed
    #end def
#end class


class ListHashingVectorizer(HashingVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', ListNGramAnalyzer(**kwargs))
        super(ListHashingVectorizer, self).__init__(**kwargs)
    #end def
#end class


class CounterHashingVectorizer(HashingVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', _counter_analyzer)
        super(CounterHashingVectorizer, self).__init__(**kwargs)
    #end def
#end class


class SpaceTokenizerTransformer(PureTransformer):
    def transform_one(self, text, **kwargs):
        return text.split()
#end class
