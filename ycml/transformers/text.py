__all__ = ['CounterHashingVectorizer', 'ListHashingVectorizer', 'ListCountVectorizer', 'SpaceTokenizerTransformer']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from .base import PureTransformer


class ListNGramAnalyzer(object):
    def __init__(self, ngram_range=(1, 1), ngram_delimiter=' ', **kwargs):
        if isinstance(ngram_range, tuple) and len(ngram_range) == 2: ngrams = list(range(ngram_range[0], ngram_range[1] + 1))
        elif isinstance(ngram_range, str): ngrams = [ngram_range]
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
    def transform_one(self, text):
        return text.split()
#end class
