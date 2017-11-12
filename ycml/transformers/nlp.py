__all__ = ['SpacyNLPProcessor', 'SpacyDocSerializer', 'NounPhraseExtractor', 'NamedEntityExtractor', 'TextRankerKeywordExtractor']

from itertools import combinations, product
import json
import logging
import os
import re

try: import networkx as nx
except ImportError: nx = None

try:
    import spacy
    from spacy import symbols
except ImportError:
    spacy = None
    symbols = None
#end try

from .base import PureTransformer

logger = logging.getLogger(__name__)


class SpacyNLPProcessor(PureTransformer):
    def __init__(self, model_name='en', spacy_args={}, use_tagger=True, use_parser=True, use_entity=True, batch_size=128, n_jobs=None, generator=False, **kwargs):
        kwargs.setdefault('nparray', False)
        super(SpacyNLPProcessor, self).__init__(**kwargs)

        if spacy is None:
            raise ImportError('You need to install the spacy NLP package.')

        if n_jobs is None: self.n_jobs = int(os.environ.get('N_JOBS', 1))
        else: self.n_jobs = n_jobs

        self.batch_size = batch_size
        self.generator = generator

        self.spacy_args = spacy_args
        self.model_name = model_name
        self.use_tagger = use_tagger
        self.use_parser = use_parser
        self.use_entity = use_entity

        self.nlp = self._setup_nlp(model_name, spacy_args, use_tagger, use_parser, use_entity)
    #end def

    def _setup_nlp(self, model_name='en', spacy_args={}, use_tagger=True, use_parser=True, use_entity=True):
        nlp = spacy.load(model_name, **spacy_args)
        if 'pipeline' not in spacy_args and 'create_pipeline' not in spacy_args:
            pipeline = []
            if use_tagger: pipeline.append(nlp.tagger)
            if use_parser: pipeline.append(nlp.parser)
            if use_entity: pipeline.append(nlp.entity)

            nlp.pipeline = pipeline
        #end if

        logger.debug('Loaded spacy model {}'.format(json.dumps(nlp.meta)))

        return nlp
    #end def

    def _transform(self, X_texts, **kwargs):
        if self.generator:
            return (doc for doc in self.nlp.pipe(X_texts, batch_size=self.batch_size, n_threads=self.n_jobs))

        return [self.nlp(text) for text in X_texts]
    #end def

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('nlp', None)

        return state
    #end def

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nlp = self._setup_nlp(self.model_name, self.spacy_args, self.use_tagger, self.use_parser, self.use_entity)
    #end def
#end def


class SpacyDocSerializer(PureTransformer):
    def __init__(self, **kwargs):
        kwargs.setdefault('nparray', False)
        super(SpacyDocSerializer, self).__init__(**kwargs)
    #end def

    def transform_one(self, doc, **kwargs):
        tokens = {}
        for tok in doc:
            tokens[tok.i] = dict(idx=tok.idx, ent_type=tok.ent_type_, ent_iob=tok.ent_iob_, lemma=tok.lemma_, text=tok.text, pos=tok.pos_, tag=tok.tag_, head=tok.head.i, dep=tok.dep_)
        #end for

        sentences = [dict(tokens=[tok.i for tok in sent]) for sent in doc.sents]
        entities = [dict(tokens=[tok.i for tok in ent], label=ent.label_) for ent in doc.ents]
        noun_chunks = [dict(tokens=[tok.i for tok in chunk]) for chunk in doc.noun_chunks]

        d = dict(length=len(doc), text=doc.text, tokens=tokens, sentences=sentences, entities=entities, noun_chunks=noun_chunks)

        return d
    #end def
#end class


class NamedEntityExtractor(PureTransformer):
    def __init__(self, entity_labels=None, **kwargs):
        kwargs.setdefault('nparray', False)
        super(NamedEntityExtractor, self).__init__(**kwargs)

        self.entity_labels = entity_labels
    #end def

    def transform_one(self, doc, **kwargs):
        entities = [ent for ent in doc.ents if self.entity_labels is None or ent.label in self.entity_labels or ent.label_ in self.entity_labels]

        return entities
    #end def
#end class


class NounPhraseExtractor(PureTransformer):
    # def __init__(self, np_rules=[r'(N|J)'], **kwargs):
    def __init__(self, np_rules=[r'((J|X|N|0)(J|C|X|0|N)*(N))'], **kwargs):
        kwargs.setdefault('nparray', False)
        super(NounPhraseExtractor, self).__init__(**kwargs)

        self.POS_TO_CHAR = {
            symbols.ADJ: 'J',
            symbols.ADP: 'P',
            symbols.ADV: 'R',
            symbols.AUX: 'A',
            symbols.CONJ: 'C',
            symbols.CCONJ: 'C',
            symbols.DET: 'D',
            symbols.INTJ: 'I',
            symbols.NOUN: 'N',
            symbols.NUM: '0',
            symbols.PART: 'T',
            symbols.PRON: 'O',
            symbols.PROPN: 'N',
            symbols.PUNCT: '.',
            symbols.SCONJ: 'C',
            symbols.SYM: '#',
            symbols.VERB: 'V',
            symbols.X: 'X',
            symbols.SPACE: '.',
            symbols.EOL: '.',
        }

        self.np_rules = []
        for rule in np_rules:
            if isinstance(rule, str): self.np_rules.append(re.compile(rule))
            else: self.np_rules.append(rule)
        #end for
    #end def

    def transform_one(self, doc, **kwargs):
        chunks = []
        for sent in doc.sents:
            pos_tags = ''.join(self.POS_TO_CHAR[tok.pos] for tok in sent)

            for rule in self.np_rules:
                for m in rule.finditer(pos_tags):

                    chunks.append(sent[m.span(1)[0]:m.span(1)[1]])
                    # print(sent[m.span(1)[0]:m.span(1)[1]], pos_tags[m.span(1)[0]:m.span(1)[1]])
                #end for
            #end for
        #end for

        return chunks
    #end def
#end class


class TextRankerKeywordExtractor(PureTransformer):
    def __init__(self, use_lemma=True, coocurrence_window_size=8, **kwargs):
        kwargs.setdefault('nparray', False)
        super(TextRankerKeywordExtractor, self).__init__(**kwargs)

        if nx is None: raise ImportError('You need to install networkx for the TextRankerKeywordExtractor.')

        self.use_lemma = use_lemma
        self.coocurrence_window_size = coocurrence_window_size
    #end def

    def transform_one(self, phrases, **kwargs):
        vocabulary = []
        vocab_map = {}

        for phrase in phrases:
            normalized = self.normalize_phrase(phrase)
            if normalized in vocab_map:
                vocab_map[normalized].append(phrase)
            else:
                vocab_map[normalized] = [phrase]
                vocabulary.append(normalized)
            #end if
        #end for

        vocabulary.sort()
        # vocab_map_index = dict((normalized, i) for i, normalized in enumerate(vocabulary))
        N = len(vocabulary)

        # Build the graph
        gr = nx.Graph()
        gr.add_nodes_from(range(N))
        for i, j in combinations(range(N), 2):
            phrases_i = vocab_map[vocabulary[i]]
            phrases_j = vocab_map[vocabulary[j]]
            distance = self.compute_cooccurrence_distance(phrases_i, phrases_j)
            if distance <= self.coocurrence_window_size:
                gr.add_edge(i, j)
        #end for

        calculated_page_rank = nx.pagerank(gr)
        keyphrase_indexes = sorted(calculated_page_rank.keys(), key=calculated_page_rank.get, reverse=True)
        keyphrases = []
        for i in keyphrase_indexes:
            keyphrases.append((vocab_map[vocabulary[i]][0], calculated_page_rank[i]))
        #end for

        return keyphrases
    #end def

    def compute_cooccurrence_distance(self, phrases_i, phrases_j):
        # for p_i, p_j in product(phrases_i, phrases_j):
        #     print(p_i, 'x', p_j, abs(p_i[0].i - p_j[0].i))

        return min(abs(p_i[0].i - p_j[0].i) for p_i, p_j in product(phrases_i, phrases_j))
    #end def

    def normalize_phrase(self, phrase):
        if self.use_lemma:
            return ' '.join(tok.lemma_ for tok in phrase)
        return ' '.join(tok.lower_ for tok in phrase)
    #end def
#end class


def main():
    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.DEBUG)

    nlp_processor = SpacyNLPProcessor()
    np_extractor = NounPhraseExtractor()
    textrank_extractor = TextRankerKeywordExtractor()

    text = """
    Bertram "Bert" Thomas Combs (August 13, 1911 – December 4, 1991) was a jurist and politician from the U.S. state of Kentucky. After serving on the Kentucky Court of Appeals, he was elected the 50th Governor of Kentucky in 1959 on his second run for the office. Following his gubernatorial term, he was appointed to the Sixth Circuit Court of Appeals by President Lyndon B. Johnson, serving from 1967 to 1970.

    Combs rose from poverty in his native Clay County to obtain a law degree from the University of Kentucky and open a law practice in Prestonsburg. He was decorated for prosecuting Japanese war criminals before military tribunals following World War II, then returned to Kentucky and his law practice. In 1951, Governor Lawrence Wetherby appointed him to fill a vacancy on the Kentucky Court of Appeals. Later that year, he was elected to a full term on the court, defeating former governor and judge Simeon S. Willis. Kentucky's Democratic Party had split into two factions by 1955 when Earle C. Clements, the leader of one faction, chose Combs to challenge former governor and U.S. Senator A. B. "Happy" Chandler, who headed the other, in the upcoming gubernatorial primary. Combs' uninspiring speeches and candidness about the need for more state revenue cost him the primary election. Chandler, who went on to reclaim the governorship, had promised that he would not need to raise taxes to meet the state's financial obligations, but ultimately he did so. This damaged Chandler's credibility and left Combs looking courageous and honest in the eyes of the electorate. Consequently, in 1959 Combs was elected governor, defeating Lieutenant Governor Harry Lee Waterfield, Chandler's choice to succeed him in office, in the primary. Early in his term, Combs secured passage of a three-percent sales tax to pay a bonus to the state's military veterans. Knowing a tax of one percent would have been sufficient, he used the excess revenue to enact a system of reforms, including expansion of the state's highway and state park systems. He also devoted much of the surplus to education.
    """

    for doc_chunks in textrank_extractor.transform(np_extractor.transform(nlp_processor.transform([text]))):
        print(doc_chunks)
#end def


if __name__ == '__main__': main()
