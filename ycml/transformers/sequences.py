__all__ = ['TokensToIndexTransformer']

import logging

from .base import PureTransformer
from .text import ListCountVectorizer

logger = logging.getLogger(__name__)


class TokensToIndexTransformer(PureTransformer):
    def __init__(self, skip_unknown=False, pad_sequences=None, count_vectorizer_args={}, pad_sequences_args={}, **kwargs):
        super(TokensToIndexTransformer, self).__init__(**kwargs)

        self.skip_unknown = skip_unknown
        self.pad_sequences = pad_sequences
        self.count_vectorizer_args = count_vectorizer_args
        self.pad_sequences_args = pad_sequences_args
    #end def

    def fit(self, X, *args, **kwargs):
        self.count_vectorizer_ = ListCountVectorizer(**self.count_vectorizer_args).fit(X)
        logger.debug('TokensToIndexTransformer vocabulary fitted with size {}.'.format(len(self.vocabulary_)))

        return self
    #end def

    def _transform(self, X, y=None, **kwargs):
        if 'maxlen' in self.pad_sequences_args:
            raise ValueError('The `maxlen` argument should not be set in `pad_sequences_args`. Set it in `pad_sequences` instead.')

        analyzer = self.count_vectorizer_.build_analyzer()
        V = self.vocabulary_

        X_transformed = []
        for seq in X:
            indexes = []
            for j, tok in enumerate(analyzer(seq)):
                index = V.get(tok)

                if not getattr(self, 'skip_unknown', False): indexes.append(0 if index is None else (index + 1))
                elif index is not None: indexes.append(index)
            #end for

            X_transformed.append(indexes)
        #end for

        if self.pad_sequences is not None:
            from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

            maxlen = getattr(self, 'pad_sequences_maxlen_', None if self.pad_sequences is True else self.pad_sequences)
            X_transformed = keras_pad_sequences(X_transformed, maxlen=maxlen, **self.pad_sequences_args)
            if self.pad_sequences is True or maxlen is not None:
                logger.debug('TokensToIndexTransformer transformed sequences has max length {}.'.format(X_transformed.shape[1]))

            self.pad_sequences_maxlen_ = X_transformed.shape[1]
        #end if

        return X_transformed
    #end def

    @property
    def vocabulary_(self): return self.count_vectorizer_.vocabulary_

    @property
    def stop_words_(self): return self.count_vectorizer_.stop_words_

    def __repr__(self):
        count_vectorizer_repr = '{}(vocabulary_={}, stop_words_={})'.format(self.count_vectorizer_.__class__.__name__, len(getattr(self.count_vectorizer_, 'vocabulary_', [])), len(getattr(self.count_vectorizer_, 'stop_words_', []))) if hasattr(self, 'count_vectorizer_') else None

        return '{}(skip_unknown={}, pad_sequences={}, count_vectorizer_args={}, pad_sequences_args={}, count_vectorizer_={})'.format(self.__class__.__name__, self.skip_unknown, self.pad_sequences, self.count_vectorizer_args, self.pad_sequences_args, count_vectorizer_repr)
    #end def
#end class
