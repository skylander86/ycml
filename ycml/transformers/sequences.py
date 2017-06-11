import logging

from .base import PureTransformer
from .text import ListCountVectorizer

__all__ = ['TokensToIndexTransformer']

logger = logging.getLogger(__name__)


class TokensToIndexTransformer(PureTransformer):
    def __init__(self, ignore_unknown=False, pad_sequences=None, count_vectorizer_args={}, pad_sequences_args={}, **kwargs):
        super(TokensToIndexTransformer, self).__init__(**kwargs)

        self.ignore_unknown = ignore_unknown
        self.pad_sequences = pad_sequences
        self.count_vectorizer_args = count_vectorizer_args
        self.pad_sequences_args = pad_sequences_args
    #end def

    def fit(self, X, **kwargs):
        self.count_vectorizer_ = ListCountVectorizer(**self.count_vectorizer_args).fit(X)
        logger.debug('TokensToIndexTransformer vocabulary fitted with size {}.'.format(len(self.vocabulary_)))

        return self
    #end def

    def _transform(self, X, y=None):
        if 'maxlen' in self.pad_sequences_args:
            raise ValueError('The `maxlen` argument should not be set in `pad_sequences_args`. Set it in `pad_sequences` instead.')

        analyzer = self.count_vectorizer_.build_analyzer()
        V = self.vocabulary_
        unknown_index = 1 if self.ignore_unknown else len(V)

        X_transformed = []
        for seq in X:
            indexes = []
            for j, tok in enumerate(analyzer(seq)):
                index = V.get(tok, unknown_index)

                if index >= 0:
                    indexes.append(index)
            #end for

            X_transformed.append(indexes)
        #end for

        if self.pad_sequences is not None:
            from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
            X_transformed = keras_pad_sequences(X_transformed, maxlen=None if self.pad_sequences is True else self.pad_sequences, **self.pad_sequences_args)
            if self.pad_sequences is True:
                logger.debug('TokensToIndexTransformer transformed sequences has max length {}.'.format(X_transformed.shape[1]))
        #end if

        return X_transformed
    #end def

    @property
    def vocabulary_(self): return self.count_vectorizer_.vocabulary_

    @property
    def stop_words_(self): return self.count_vectorizer_.stop_words_
#end class
