import spacy
from .deps import TripleExtraction_Deps
from .parseTree import TripleExtraction
from .Constants import Constants

nlp_german = spacy.load("de_core_news_sm")
nlp_french = spacy.load("fr_core_news_sm")
nlp_spanish = spacy.load("es_core_news_sm")

class SpacyDependencyParser(object):
    def __init__(self, nlp_model):
        self.nlp_model = nlp_model


    def get_dependencies(self, sentence):
        """
        The dependency format must be 
            (('word', POS), dependency, ('word', POS))
        the first dependency is 
        """
        doc = self.nlp_model(sentence)
        dependencies_list = list()
        for token in doc:
            dep = ((token.head.text, token.head.tag_), token.dep_, (token.text, token.tag_))
            dependencies_list.append(dep)
        return dependencies_list


class GermanDependencyParse(SpacyDependencyParser):
    def __init__(self):
        super().__init__(nlp_german)

class FrenchDependencyParse(SpacyDependencyParser):
    def __init__(self):
        super().__init__(nlp_french)

class SpanishDependencyParse(SpacyDependencyParser):
    def __init__(self):
        super().__init__(nlp_spanish)

class TripleExtraction_Deps_Lang(TripleExtraction_Deps):
    def __init__(self, language):
        super().__init__()
        self.dep_parser = None
        if language.lower() == "german":
            self.dep_parser = GermanDependencyParse()
        elif language.lower() == 'french':
            self.dep_parser = FrenchDependencyParse()
        elif language.lower() == "spanish":
            self.dep_parser = SpanishDependencyParse()
        else:
            raise Exception("Given language is not supported or does not exist. Only english, french, german, spanish are supported")

        def dependency_triplets(self, sentence):
            word_tokenized_sent = word_tokenize(sentence)
            dependencies = self.dep_parser.get_dependencies(sentence)
            return dependencies 


class TripleExtraction_Lang(TripleExtraction):
    def __init__(self, language):
        super().__init__()
        self.language = language.lower()
        self.parser = None
        if language == 'german':
            self.parser = CoreNLPParser(url='http://localhost:{}'.fornat(Constants.german_port))
        elif language == 'french':
            self.parser = CoreNLPParser(url='http://localhost:{}'.fornat(Constants.french_port))
        else:
            raise Exception("Given language is not supported or does not exist. Only english, french, german, spanish are supported")

    def treebank(self, sentence):
        tree = list(self.parser.parse(sentence.split()))[0]
        triple = self.main(ParentedTree.convert(tree))
        return triple

 