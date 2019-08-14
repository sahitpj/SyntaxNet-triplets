from .deps import TripleExtraction_Deps
from .Utils import hearst_get_triplet, hypernym_clean, directRelation_clean, short_relations_clean, annotate_triple
import spacy
import neuralcoref
from nltk.tokenize import sent_tokenize

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

class AdvancedTripleExtractionDeps(TripleExtraction_Deps):
    def __init__(self, text, filepath_to_conll=None, deps_level=None):
        super().__init__(filepath_to_conll=None, deps_level=None)
        self.text = text
        doc = nlp(self.text)
        self.coref_fixed_text = doc._.coref_resolved

    def get_dependencies(self, sentence):
        """
        The dependency format must be 
            (('word', POS), dependency, ('word', POS))
        the first dependency is 
        """
        doc = nlp(sentence)
        dependencies_list = list()
        for token in doc:
            dep = ((token.head.text, token.head.tag_), token.dep_, (token.text, token.tag_))
            dependencies_list.append(dep)
        return dependencies_list

    def get_entites(self, sentence):
        doc = nlp(sentence)
        return [ ent.text for ent in doc.ents ]

    def tripletsEntityCheck(self, triplets, sentence):
        """
        Replaces main words with the entire entites they reperesent
        """
        entity_triples = list()
        entities = self.get_entites(sentence)
        for triple in triplets:
            triple = list(triple)
            for entity in entities:
                if triple[0] in entity:
                    triple[0] = entity
                    break
            for entity in entities:
                if triple[2] in entity:
                    triple[2] = entity
                    break
            if triple[0] != triple[2]:
                entity_triples.append(triple)
        return entity_triples
    
    
    def get_triples(self):
        NOUN_RELATIONS = ['nmod', 'hypernym (low confidence)']
        triplets = list()
        cleaned_hypernyms = list()
        for sentence in sent_tokenize(self.coref_fixed_text):
            sentence_triples = list()
            dependencies = self.dependency_triplets(sentence)
            direct_relations, short_relations, hypernyms, prepositions = self.short_relations(dependencies, 2)
            cleaned_hypernyms.extend([ hypernym_clean(hypernym) for hypernym in hypernyms ])
            cleaned_drs = [ directRelation_clean(direct_relation) for direct_relation in direct_relations ]
            for i in cleaned_drs:
                if i[1] in NOUN_RELATIONS:
                    sentence_triples.append(i)
            for short_relation in range(len(short_relations)):
                sentence_triples.extend(short_relations_clean(short_relations[short_relation], prepositions[short_relation]))
            triplets += self.tripletsEntityCheck(sentence_triples, sentence)
            triplets += self.tripletsEntityCheck(cleaned_hypernyms, sentence)
        return triplets
        