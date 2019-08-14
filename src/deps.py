import sys
sys.path.append("../../..")
sys.path.append("..")

from .conllu.conllu import parse_single, TokenList

from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree, Tree
from nltk.tokenize import word_tokenize
from .Constants import Constants
import json


parser = CoreNLPParser(url='http://localhost:9000')
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

class TripleExtraction_Deps(object):

    def __init__(self, filepath_to_conll=None, deps_level=None):
        self.deps_level = deps_level
        self.filepath_to_conll = None
        self.tokenlist = None
        self.tokenTree = None
        if filepath_to_conll:
            self.filepath_to_conll = filepath_to_conll
            data_file = open(filepath_to_conll, "r", encoding="utf-8")
            self.tokenlist = parse_single(data_file)
            self.tokenTree = tokenlist[0].to_tree()

        self.Constants = Constants()

    def dependency_triplets(self, sentence):
        word_tokenized_sent = word_tokenize(sentence)
        parses = dep_parser.parse(word_tokenized_sent)
        dependencies = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses][0]
        return dependencies

    def bfs_triplets(self, start_dep, level, dependencies):
        '''
        finds all noun-dependencies from a start_dep

        returns connected dependencies (which are not nouns), connected noun dependencies, within the level limit
        '''
        node_1 = start_dep[0]
        node_2 = start_dep[2]
        count = 0
        queue = [node_2]
        queue_level = [0]
        connected_dependencies = list()
        connected_noun_dependencies = list()
        while queue != []:
            node = queue[0]
            queue = queue[:-1]
            level_current = queue_level[0]
            queue_level = queue_level[:-1]
            for dep in dependencies:
                dep_node_1 = dep[0]
                if dep_node_1 == node and level_current < level:
                    if dep[2][1] not in self.Constants.NOUNS:
                        queue.append(dep[2])
                        queue_level.append(level_current+1)
                        connected_dependencies.append(dep)
                    else:
                        connected_noun_dependencies.append(dep)
        return (connected_dependencies, connected_noun_dependencies)


    
    def short_relations(self, dependencies, width):
        '''
        width is the number of nodes between the source and destination

        returns the 
            direct relations, these are immediate but not with the nsubj relation
            short relations which are the short relations between nouns and connections
            hypernyms - direct relations which have the relation of nsubj
        '''
        hypernyms = list()
        direct_relations = list()
        short_relations = [] #a list of double tuples, first one with the start dep and the second one as the short relations
        for connection in dependencies:
            node_1 = connection[0]
            node_2 = connection[2]
            if node_1[1] in self.Constants.NOUNS and node_2[1] in self.Constants.NOUNS:
                if connection[1] == 'nsubj':
                    hypernyms.append(connection)
                else:
                    direct_relations.append(connection)
            elif node_1[1] in self.Constants.NOUNS:
                r = self.bfs_triplets(connection, width, dependencies)[1] #checks for noun dependencies which are indirect 
                if len(r) >=1 :
                    short_relations.append([node_1, r])
        prepositions = list()
        for r in short_relations:
            n2 = r[1][-1][-1]
            prepositions_list = self.get_prepositions(n2, dependencies)
            prepositions.append(prepositions_list)
        return direct_relations, short_relations, hypernyms, prepositions

    def get_prepositions(self, start_node, dependencies):
        count = 0
        queue = [start_node]
        queue_level = [0]
        preposition_dependencies = list()
        while queue != []:
            node = queue[0]
            queue = queue[:-1]
            level_current = queue_level[0]
            queue_level = queue_level[:-1]
            for dep in dependencies:
                dep_node_1 = dep[0]
                if dep_node_1 == node and level_current < 1:
                    if dep[1] in self.Constants.preposition_relations:
                        queue.append(dep[2])
                        queue_level.append(level_current+1)
                        preposition_dependencies.append(dep)
        print(preposition_dependencies)
        return preposition_dependencies        


class TripleExtraction_Deps_SS(TripleExtraction_Deps):
    def __init__(self, filepath_to_conll=None, deps_level=None):
        super().__init__(filepath_to_conll=None, deps_level=None)


if __name__=="__main__" :
    import sys   
    # Parse the example sentence
    sent = "Astatine is a radioactive chemical element with the chemical symbol At and atomic number 85, and is the rarest naturally occurring element on the Earth's crust."
    t = TripleExtraction()
    deps = t.dependency_triplets(sent)
    print(t.short_relations(deps, 2)[2])
