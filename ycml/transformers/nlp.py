__all__ = ['SpacyNLPProcessor', 'NounPhraseExtractor']

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
except ImportError: spacy = None

from .base import PureTransformer

logger = logging.getLogger(__name__)


class SpacyNLPProcessor(PureTransformer):
    def __init__(self, model_name='en', spacy_args={}, use_tagger=True, use_parser=True, use_entity=True, batch_size=128, n_jobs=None, generator=False, **kwargs):
        nparray = kwargs.pop('nparray', False)
        super(SpacyNLPProcessor, self).__init__(nparray=nparray, **kwargs)

        if spacy is None:
            raise ImportError('You need to install the spacy NLP package.')

        if n_jobs is None: self.n_jobs = os.environ.get('N_JOBS', 1)
        else: self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.generator = generator

        self.nlp = spacy.load(model_name, **spacy_args)
        if 'pipeline' not in spacy_args and 'create_pipeline' not in spacy_args:
            pipeline = []
            if use_tagger: pipeline.append(self.nlp.tagger)
            if use_parser: pipeline.append(self.nlp.parser)
            if use_entity: pipeline.append(self.nlp.entity)

            self.nlp.pipeline = pipeline
        #end if

        logger.debug('Loaded spacy model {}'.format(json.dumps(self.nlp.meta)))
    #end def

    def _transform(self, X_texts, **kwargs):
        if self.generator: return (doc for doc in self.nlp.pipe(X_texts, batch_size=self.batch_size, n_threads=self.n_jobs))
        return [doc for doc in self.nlp.pipe(X_texts, batch_size=self.batch_size, n_threads=self.n_jobs)]
    #end def
#end def


class NounPhraseExtractor(PureTransformer):
    POS_TO_CHAR = {
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

    # def __init__(self, np_rules=[r'(N|J)'], **kwargs):
    def __init__(self, np_rules=[r'((J|X|N|0)(J|C|X|0|N)*(N|0))'], **kwargs):
        nparray = kwargs.pop('nparray', False)
        super(NounPhraseExtractor, self).__init__(nparray=nparray, **kwargs)

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
        nparray = kwargs.pop('nparray', False)
        super(TextRankerKeywordExtractor, self).__init__(nparray=nparray, **kwargs)

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
        keyphrase_indexes = sorted(calculated_page_rank.keys(), key=calculated_page_rank.get, reverse=True)[:int(N / 3) + 1]
        keyphrases = []
        for i in keyphrase_indexes:
            print(vocabulary[i], calculated_page_rank[i])
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
