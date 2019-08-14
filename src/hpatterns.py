import sys
sys.path.append("../../..")

import re
import string
import spacy
from .conllu.conllu import parse_single, TokenList
from .hpatternUtils import create_default, create_greedy, create_semi

class HearstPatterns(object):
    """
    Contains two methods. One which uses the .conllu file to develop a tokentree which can then 
    be converted into a tagged sentence to be able to extract hearst pattern,
    the second one uses Spacy to derive the tagged sentence to be able to extract hearst patterns.

    For tagged sentences, check out the get_noun_chunks functions.
    """

    def __init__(self, extended = False, greedy = False, same_sentence = False, semi = False):
        self.__adj_stopwords = ['able', 'available', 'brief', 'certain', 'different', 'due', 'enough', 'especially','few', 'fifth', 'former', 'his', 'howbeit', 'immediate', 'important', 'inc', 'its', 'last', 'latter', 'least', 'less', 'likely', 'little', 'many', 'ml', 'more', 'most', 'much', 'my', 'necessary', 'new', 'next', 'non', 'old', 'other', 'our', 'ours', 'own', 'particular', 'past', 'possible', 'present', 'proud', 'recent', 'same', 'several', 'significant', 'similar', 'such', 'sup', 'sure']
        # now define the Hearst patterns
        # format is <hearst-pattern>, <general-term>
        # so, what this means is that if you apply the first pattern, the firsr Noun Phrase (NP)
        # is the general one, and the rest are specific NPs
        self.__hearst_patterns = [
            ('(NP_\\w+ (, )?such as (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(such NP_\\w+ (, )?as (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last', 'typeOf', 0),
            ('(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            (r'NP_(\w+).*?born.*on.* CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'bornOn', 4),
            (r'NP_(\w+).*?(died|passed away).*?on.*?CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'diedOn', 4),
            (r'NP_(\w+).*?(born|developed|made|established|published).*?(in|at).*?CD_(\w+)', 'last', 'madeIn', 4),
            (r'NP_(\w+).*?(present|found).*?in.*?NP_(\w+)', 'last', 'foundIn', 3),
            (r'NP_(\w+).*?(member).*?of.*?NP_(\w+)', 'last', 'memberOf', 3),
            (r'NP_(\w+).*?(developed|made|published|established).*?by.*?NP_(\w+)', 'last', 'madeBy', 3),
            (r'NP_(\w+).*?(composed).*?of.*?NP_(\w+)', 'last', 'composedOf', 3),
            (r'NP_(\w+).*?also known as.*?NP_(\w+)', 'last', 'also known as', 2),
            (r'NP_(\w+).*?located.*?(in|on).*?NP_(\w+)', 'last', 'locatedIn|On', 3),
            (r'NP_(\w+).*?(was|is) a.*?NP_(\w+)', 'first', 'attribute', 3),
            (r'NP_(\w+).*?(comparable|related) to.*?NP_(\w+)', 'last', 'comparable to', 3),
            ('(NP_\\w+ (, )?made of (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'madeOf_multiple', 0),
            ('(NP_\\w+ (, )?(was|is) a (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'attribute|profession_multiple', 0),
            (r'NP_(\w+).*?(was|is).*?published.*?(in|on).*?CD_(\w+)', 'first', 'publishedIn', 3),
            (r'NP_(\w+).*?represent.*?NP_(\w+)', 'first', 'representedBy', 2),
            (r'NP_(\w+).*?used.*?(by|in|as).*?NP_(\w+)', 'first', 'used_', 3),
            (r'NP_(\w+).*?made.*?of.*?NP_(\w+)', 'first', 'madeOf', 2),
            (r'NP_(\w+).*?form.*?of.*?NP_(\w+)', 'first', 'formOf', 2),
            (r'NP_(\w+).*?(leader|ruler|king|head).*?of.*?NP_(\w+)', 'first', 'leaderOf', 3),
            (r'NP_(\w+).*?famous.*?for.*?NP_(\w+)', 'first', 'famousFor', 2),
            ('(NP_\\w+ (, )?famous for (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'FamousFor_multiple', 0),
        ] + create_default()

        self.__hearst_patterns_greedy = [
            ('(NP_\\w+ (, )?such as (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(such NP_\\w+ (, )?as (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last', 'typeOf', 0),
            ('(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            (r'.*NP_(\w+).*?born.*on.* CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'bornOn', 4),
            (r'.*NP_(\w+).*?(died|passed away).*?on.*?CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'diedOn', 4),
            (r'.*NP_(\w+).*?(born|developed|made|established|published).*?(in|at).*?CD_(\w+)', 'last', 'madeIn', 4),
            (r'.*NP_(\w+).*?(present|found).*?in.*?NP_(\w+)', 'last', 'foundIn', 3),
            (r'.*NP_(\w+).*?(member).*?of.*?NP_(\w+)', 'last', 'memberOf', 3),
            (r'.*NP_(\w+).*?(developed|made|published|established).*?by.*?NP_(\w+)', 'last', 'madeBy', 3),
            (r'.*NP_(\w+).*?(composed).*?of.*?NP_(\w+)', 'last', 'composedOf', 3),
            (r'.*NP_(\w+).*?also known as.*?NP_(\w+)', 'last', 'also known as', 2),
            (r'.*NP_(\w+).*?located.*?(in|on).*?NP_(\w+)', 'last', 'locatedIn|On', 3),
            (r'.*NP_(\w+).*?(was|is) a.*?NP_(\w+)', 'first', 'attribute', 3),
            (r'.*NP_(\w+).*?(comparable|related) to.*?NP_(\w+)', 'last', 'comparable to', 3),
            ('(NP_\\w+ (, )?made of (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'madeOf_multiple', 0),
            ('(NP_\\w+ (, )?(was|is) a (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'attribute|profession_multiple', 0),
            (r'.*NP_(\w+) (was|is).*?published.*?(in|on).*?CD_(\w+)', 'first', 'publishedIn', 3),
            (r'.*NP_(\w+).*?represent.*?NP_(\w+)', 'first', 'representedBy', 2),
            (r'.*NP_(\w+).*?used.*?(by|in|as).*?NP_(\w+)', 'first', 'used_', 3),
            (r'.*NP_(\w+).*?made.*?of.*?NP_(\w+)', 'first', 'madeOf', 2),
            (r'.*NP_(\w+).*?form.*?of.*?NP_(\w+)', 'first', 'formOf', 2),
            (r'.*NP_(\w+).*?(leader|ruler|king|head) .*?of.*?NP_(\w+)', 'first', 'leaderOf', 3),
            (r'.*NP_(\w+).*?famous.*?for.*?NP_(\w+)', 'first', 'famousFor', 2),
            ('(NP_\\w+ (, )?famous for (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'FamousFor_multiple', 0),
        ]

        self.__hearst_patterns_semigreedy = [
            ('(NP_\\w+ (, )?such as (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(such NP_\\w+ (, )?as (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last', 'typeOf', 0),
            ('(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            (r'.*?NP_(\w+).*?born.*on.* CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'bornOn', 4),
            (r'.*?NP_(\w+).*?(died|passed away).*?on.*?CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'diedOn', 4),
            (r'.*?NP_(\w+).*?(born|developed|made|established|published).*?(in|at).*?CD_(\w+)', 'last', 'madeIn', 4),
            (r'.*?NP_(\w+).*?(present|found).*?in.*?NP_(\w+)', 'last', 'foundIn', 3),
            (r'.*?NP_(\w+).*?(member).*?of.*?NP_(\w+)', 'last', 'memberOf', 3),
            (r'.*?NP_(\w+).*?(developed|made|published|established).*?by.*?NP_(\w+)', 'last', 'madeBy', 3),
            (r'.*?NP_(\w+).*?(composed).*?of.*?NP_(\w+)', 'last', 'composedOf', 3),
            (r'.*?NP_(\w+).*?also known as.*?NP_(\w+)', 'last', 'also known as', 2),
            (r'.*?NP_(\w+).*?located.*?(in|on).*?NP_(\w+)', 'last', 'locatedIn|On', 3),
            (r'.*?NP_(\w+).*?(was|is) a.*?NP_(\w+)', 'first', 'attribute', 3),
            (r'.*?NP_(\w+).*?(comparable|related) to.*?NP_(\w+)', 'last', 'comparable to', 3),
            ('(NP_\\w+ (, )?made of (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'madeOf_multiple', 0),
            ('(NP_\\w+ (, )?(was|is) a (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'attribute|profession_multiple', 0),
            (r'.*?NP_(\w+) (was|is).*?published.*?(in|on).*?CD_(\w+)', 'first', 'publishedIn', 3),
            (r'.*?NP_(\w+).*?represent.*?NP_(\w+)', 'first', 'representedBy', 2),
            (r'.*?NP_(\w+).*?used.*?(by|in|as).*?NP_(\w+)', 'first', 'used_', 3),
            (r'.*?NP_(\w+).*?made.*?of.*?NP_(\w+)', 'first', 'madeOf', 2),
            (r'.*?NP_(\w+).*?form.*?of.*?NP_(\w+)', 'first', 'formOf', 2),
            (r'.*?NP_(\w+).*?(leader|ruler|king|head).*?of.*?NP_(\w+)', 'first', 'leaderOf', 3),
            (r'.*?NP_(\w+).*?famous.*?for.*?NP_(\w+)', 'first', 'famousFor', 2),
            ('(NP_\\w+ (, )?famous for (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'FamousFor_multiple', 0),
        ]


        self.__hearst_patterns_ss = [
            ('(NP_\\w+ (, )?such as (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(such NP_\\w+ (, )?as (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last', 'typeOf', 0),
            ('(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            ('(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first', 'typeOf', 0),
            (r'NP_(\w+)[^.]*?born[^.]*on[^.]* CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'bornOn', 4),
            (r'NP_(\w+)[^.]*?(died|passed away)[^.]*?on[^.]*?CD_(\d+)? (\w+) CD_(\d+)? ', 'last', 'diedOn', 4),
            (r'NP_(\w+)[^.]*?(born|developed|made|established|published)[^.]*?(in|at)[^.]*?CD_(\w+)', 'last', 'madeIn', 4),
            (r'NP_(\w+)[^.]*?(present|found)[^.]*?in[^.]*?NP_(\w+)', 'last', 'foundIn', 3),
            (r'NP_(\w+)[^.]*?(member)[^.]*?of[^.]*?NP_(\w+)', 'last', 'memberOf', 3),
            (r'NP_(\w+)[^.]*?(developed|made|published|established)[^.]*?by[^.]*?NP_(\w+)', 'last', 'madeBy', 3),
            (r'NP_(\w+)[^.]*?(composed)[^.]*?of[^.]*?NP_(\w+)', 'last', 'composedOf', 3),
            (r'NP_(\w+)[^.]*?also known as[^.]*?NP_(\w+)', 'last', 'also known as', 2),
            (r'NP_(\w+)[^.]*?located[^.]*?(in|on)[^.]*?NP_(\w+)', 'last', 'locatedIn|On', 3),
            (r'NP_(\w+)[^.]*?(was|is) a[^.]*?NP_(\w+)', 'first', 'attribute', 3),
            (r'NP_(\w+)[^.]*?(comparable|related) to[^.]*?NP_(\w+)', 'last', 'comparable to', 3),
            ('(NP_\\w+ (, )?made of (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'madeOf_multiple', 0),
            ('(NP_\\w+ (, )?(was|is) a (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'attribute|profession_multiple', 0),
            (r'NP_(\w+) (was|is)[^.]*?published[^.]*?(in|on)[^.]*?CD_(\w+)', 'first', 'publishedIn', 3),
            (r'NP_(\w+)[^.]*?represent[^.]*?NP_(\w+)', 'first', 'representedBy', 2),
            (r'NP_(\w+)[^.]*?used[^.]*?(by|in|as)[^.]*?NP_(\w+)', 'first', 'used_', 3),
            (r'NP_(\w+)[^.]*?made[^.]*?of[^.]*?NP_(\w+)', 'first', 'madeOf', 2),
            (r'NP_(\w+)[^.]*?form[^.]*?of[^.]*?NP_(\w+)', 'first', 'formOf', 2),
            (r'NP_(\w+)[^.]*?(leader|ruler|king|head)[^.]*?of[^.]*?NP_(\w+)', 'first', 'leaderOf', 3),
            (r'NP_(\w+)[^.]*?famous[^.]*?for[^.]*?NP_(\w+)', 'first', 'famousFor', 2),
            ('(NP_\\w+ (, )?famous for (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'Famousfor_multiple', 0),
        ]

        if extended:
            self.__hearst_patterns.extend([
                ('((NP_\\w+ ?(, )?)+(and |or )?any other NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?some other NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?be a NP_\\w+)', 'last', 'typeOf', 0), 
                ('(NP_\\w+ (, )?like (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('such (NP_\\w+ (, )?as (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?like other NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?one of the NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?one of these NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?one of those NP_\\w+)', 'last', 'typeOf', 0),
                ('example of (NP_\\w+ (, )?be (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?be example of NP_\\w+)', 'last', 'typeOf', 0),
                ('(NP_\\w+ (, )?for example (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?wich be call NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?which be name NP_\\w+)', 'last', 'typeOf', 0),
                ('(NP_\\w+ (, )?mainly (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?mostly (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?notably (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?particularly (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?principally (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?in particular (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?except (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?other than (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?e.g. (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?i.e. (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?a kind of NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?kind of NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?form of NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?which look like NP_\\w+)', 'last', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?which sound like NP_\\w+)', 'last', 'typeOf', 0),
                ('(NP_\\w+ (, )?which be similar to (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?example of this be (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?type (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )? NP_\\w+ type)', 'last', 'typeOf', 0),
                ('(NP_\\w+ (, )?whether (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(compare (NP_\\w+ ?(, )?)+(and |or )?with NP_\\w+)', 'last', 'typeOf', 0),
                ('(NP_\\w+ (, )?compare to (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('(NP_\\w+ (, )?among -PRON- (NP_\\w+ ? (, )?(and |or )?)+)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?as NP_\\w+)', 'last', 'typeOf', 0),
                ('(NP_\\w+ (, )? (NP_\\w+ ? (, )?(and |or )?)+ for instance)', 'first', 'typeOf', 0),
                ('((NP_\\w+ ?(, )?)+(and |or )?sort of NP_\\w+)', 'last', 'typeOf', 0)
            ])

        self.__spacy_nlp = spacy.load('en')

        if greedy:
            self.__hearst_patterns = self.__hearst_patterns_greedy + create_greedy()

        if same_sentence:
            self.__hearst_patterns = self.__hearst_patterns_ss

        if semi:
            self.__hearst_patterns = self.__hearst_patterns_semigreedy + create_semi()
        
    def chunk(self, rawtext):
        STOP_TOKENS = ["the", "a", "an"]
        doc = self.__spacy_nlp(rawtext)
        chunks = []
        for sentence in doc.sents:
            sentence_text = sentence.text.lower()
            for chunk in sentence.noun_chunks:
                chunk_arr = []
                replace_arr = []
                for token in chunk:
                    if token.text not in STOP_TOKENS:
                        chunk_arr.append(token.text)
                    # Remove punctuation and stopword adjectives (generally quantifiers of plurals)
                    if token.lemma_.isalnum() and token.lemma_ not in self.__adj_stopwords and token.text not in STOP_TOKENS:
                        replace_arr.append(token.lemma_)
                    elif not token.lemma_.isalnum() and token.text not in STOP_TOKENS:
                        if token.lemma_ != '-PRON-':
                            replace_arr.append(''.join(char for char in token.lemma_ if char.isalnum()))
                        else:
                            replace_arr.append(token.text)
                chunk_lemma = ' '.join(chunk_arr).lower()
                replacement_value = 'NP_' + '_'.join(replace_arr).lower()
                if chunk_lemma:
                    sentence_text = re.sub(r'\b%s\b' % re.escape(chunk_lemma),
                                           r'%s' % replacement_value,
                                           sentence_text)
            chunks.append(sentence_text)
        return chunks

    def chunk_root(self, rawtext):
        doc = self.__spacy_nlp(rawtext)
        chunks = []
        for sentence in doc.sents:
            sentence_text = sentence.lemma_
            for chunk in sentence.noun_chunks:
                chunk_arr = []
                replace_arr = []
                for token in chunk:
                    chunk_arr.append(token.lemma_)
                    # Remove punctuation and stopword adjectives (generally quantifiers of plurals)
                    if token.lemma_.isalnum() and token.lemma_ not in self.__adj_stopwords:
                        replace_arr.append(token.lemma_)
                    elif not token.lemma_.isalnum():
                        replace_arr.append(''.join(char for char in token.lemma_ if char.isalnum()))
                chunk_lemma = ' '.join(chunk_arr)
                replacement_value = 'NP_' + '_'.join(replace_arr)
                if chunk_lemma:
                    sentence_text = re.sub(r'\b%s\b' % re.escape(chunk_lemma),
                                           r'%s' % replacement_value,
                                           sentence_text)
            chunks.append(sentence_text)
        return chunks

    """
        This is the main entry point for this code.
        It takes as input the rawtext to process and returns a list of tuples (specific-term, general-term)
        where each tuple represents a hypernym pair.
    """
    def find_hearstpatterns(self, filepath_to_conll, subject):
        
        data_file = open(filepath_to_conll, "r", encoding="utf-8")
        tokenList = parse_single(data_file)
        sentence_tokenList = tokenList[0]
        hearst_patterns = []
        # np_tagged_sentences = self.chunk(rawtext)
        np_tagged_sentences = sentence_tokenList.get_noun_chunks(subject)
        # for sentence in np_tagged_sentences:
        # two or more NPs next to each other should be merged into a single NP, it's a chunk error

        for (hearst_pattern, parser, hearst_type, process_type) in self.__hearst_patterns:
            matches = re.search(hearst_pattern, np_tagged_sentences)
            if matches:
                match_str = matches.group(0)

                if process_type == 0:
                    nps = [a for a in match_str.split() if a.startswith("NP_")]

                    if parser == "first":
                        general = nps[0]
                        specifics = nps[1:]
                    else:
                        general = nps[-1]
                        specifics = nps[:-1]

                    for i in range(len(specifics)):
                        #print("%s, %s %s" % (specifics[i], general, hearst_type))
                        hearst_patterns.append((self.clean_hyponym_term(specifics[i]), self.clean_hyponym_term(general), hearst_type))

                else:
                    if parser == "first":
                        general = matches.group(1)
                        specifics = [matches.group(i) for i in range(2,process_type+1)]
                    else:
                        general = matches.group(process_type)
                        specifics = [matches.group(i) for i in range(1,process_type)]

                    #print("%s, %s %s" % (specifics[i], general, hearst_type))
                    hearst_patterns.append((specifics, general, hearst_type))



        return hearst_patterns

    def find_hearstpatterns_spacy(self, rawtext):

        hearst_patterns = []
        np_tagged_sentences = self.chunk(rawtext)

        for sentence in np_tagged_sentences:
            # two or more NPs next to each other should be merged into a single NP, it's a chunk error
            for (hearst_pattern, parser, hearst_type, process_type) in self.__hearst_patterns[:-1]:
                matches = re.search(hearst_pattern, sentence)
                if matches:
                    match_str = matches.group(0)

                    if process_type == 0:
                        nps = [a for a in match_str.split() if a.startswith("NP_")]

                        if parser == "first":
                            general = nps[0]
                            specifics = nps[1:]
                        else:
                            general = nps[-1]
                            specifics = nps[:-1]

                        for i in range(len(specifics)):
                            #print("%s, %s %s" % (specifics[i], general, hearst_type))
                            hearst_patterns.append((self.clean_hyponym_term(specifics[i]), self.clean_hyponym_term(general), hearst_type))

                    else:
                        if parser == "first":
                            general = matches.group(1)
                            specifics = [matches.group(i) for i in range(2,process_type+1)]
                        else:
                            general = matches.group(process_type)
                            specifics = [matches.group(i) for i in range(1,process_type)]

                        #print("%s, %s %s" % (specifics[i], general, hearst_type))
                        hearst_patterns.append((specifics, general, hearst_type, parser))

        return hearst_patterns

    def find_hearstpatterns_spacy_root(self, rawtext):

        hearst_patterns = []
        np_tagged_sentences = self.chunk_root(rawtext)

        for sentence in np_tagged_sentences:
            # two or more NPs next to each other should be merged into a single NP, it's a chunk error
            for (hearst_pattern, parser, hearst_type, process_type) in self.__hearst_patterns:
                matches = re.search(hearst_pattern, sentence)
                if matches:
                    match_str = matches.group(0)

                    if process_type == 0:
                        nps = [a for a in match_str.split() if a.startswith("NP_")]

                        if parser == "first":
                            general = nps[0]
                            specifics = nps[1:]
                        else:
                            general = nps[-1]
                            specifics = nps[:-1]

                        for i in range(len(specifics)):
                            #print("%s, %s %s" % (specifics[i], general, hearst_type))
                            hearst_patterns.append((self.clean_hyponym_term(specifics[i]), self.clean_hyponym_term(general), hearst_type))

                    else:
                        if parser == "first":
                            general = matches.group(1)
                            specifics = [matches.group(i) for i in range(2,process_type+1)]
                        else:
                            general = matches.group(process_type)
                            specifics = [matches.group(i) for i in range(1,process_type)]

                        #print("%s, %s %s" % (specifics[i], general, hearst_type))
                        hearst_patterns.append((specifics, general, hearst_type, parser))

        return hearst_patterns

    def add_patterns(self, patterns, t):
        if t == 'Default':
            self.__hearst_patterns.extend(patterns)
        elif t == 'Non-greedy':
            self.__hearst_patterns_greedy.extend(patterns)
        else:
            self.__hearst_patterns_semigreedy.extend(patterns)


    def clean_hyponym_term(self, term):
        # good point to do the stemming or lemmatization
        return term.replace("NP_","").replace("_", " ")

