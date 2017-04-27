from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

__all__ = ['CounterHashingVectorizer', 'ListHashingVectorizer', 'ListCountVectorizer']


def _list_analyzer(L):
    for elem in L:
        yield elem
#end def


def _counter_analyzer(C):
    for k, v in C.items():
        for _ in range(v):
            yield k
#end def


class ListCountVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', _list_analyzer)
        super(ListCountVectorizer, self).__init__(**kwargs)
    #end def
#end class


class ListHashingVectorizer(HashingVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', _list_analyzer)
        super(ListHashingVectorizer, self).__init__(**kwargs)
    #end def
#end class


class CounterHashingVectorizer(HashingVectorizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', _counter_analyzer)
        super(CounterHashingVectorizer, self).__init__(**kwargs)
    #end def
#end class
