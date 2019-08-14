import sys
sys.path.append("../../..")
sys.path.append("..")

from .conllu.conllu import parse_single, TokenList
from .stanford import treegex_api

# data_file = open("sample.conll", "r", encoding="utf-8")
# tokenlist = parse_single(data_file) #tokenlist gives the parsed conllu file
# print(tokenlist[0].to_tree())
import os
#Set standford parser and models in your environment variables.
from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree, Tree
parser = CoreNLPParser(url='http://localhost:9000')


class TripleExtraction(object):
    """
    Stanford treegex utility for extracting triplets. Utilises the stanford coreNLP API for execution. Make sure that it
    is running while running. 
    """

    VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

    def __init__(self, filepath_to_conll=None):
        self.filepath_to_conll = None
        self.tokenlist = None
        self.tokenTree = None
        if filepath_to_conll:
            self.filepath_to_conll = filepath_to_conll
            data_file = open(filepath_to_conll, "r", encoding="utf-8")
            self.tokenlist = parse_single(data_file)
            self.tokenTree = tokenlist[0].to_tree()


        self.treegex_patterns = [

        ]

    def treegex(self, sentence):
        '''
        make sure the coreNLP server is running
        the default url is set to - http://localhost:9000/tregex, if this is not the url which you are using set the required url
        in the param url, and pass it to the treegex_api function (third parameter)
        '''

        responses = treegex_api(self.treegex_patterns, sentence)

    def find_subject(self, t): 
        for s in t.subtrees(lambda t: t.label() == 'NP'):
            for n in s.subtrees(lambda n: n.label().startswith('NN')):
                return (n[0], self.find_attrs(n))
                # return n[0]
                
    def find_predicate(self, t):    
        v = None
        
        for s in t.subtrees(lambda t: t.label() == 'VP'):
            for n in s.subtrees(lambda n: n.label().startswith('VB')):
                v = n
            return (v[0], self.find_attrs(v))
            # return v[0]
        
    def find_object(self,t):    
        for s in t.subtrees(lambda t: t.label() == 'VP'):
            for n in s.subtrees(lambda n: n.label() in ['NP', 'PP', 'ADJP']):
                if n.label() in ['NP', 'PP']:
                    for c in n.subtrees(lambda c: c.label().startswith('NN')):
                        return (c[0], self.find_attrs(c))
                        # return c[0]
                else:
                    for c in n.subtrees(lambda c: c.label().startswith('JJ')):
                        return (c[0], self.find_attrs(c))
                        # return c[0]
                    
    def find_attrs(self, node):
        attrs = []
        p = node.parent()
        
        # Search siblings
        if node.label().startswith('JJ'):
            for s in p:
                if s.label() == 'RB':
                    attrs.append(s[0])
                    
        elif node.label().startswith('NN'):
            for s in p:
                if s.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                    attrs.append(' '.join(s.flatten()))
        
        elif node.label().startswith('VB'):
            for s in p:
                if s.label() == 'ADVP':
                    attrs.append(' '.join(s.flatten()))
                    
        # Search uncles
        if node.label().startswith('JJ') or node.label().startswith('NN'):
            for s in p.parent():
                if s != p and s.label() == 'PP':
                    attrs.append(' '.join(s.flatten()))
                    
        elif node.label().startswith('VB'):
            for s in p.parent():
                if s != p and s.label().startswith('VB'):
                    attrs.append(s[0])
                    
        return attrs

    def main(self, sentence):
        try:
            subject = self.find_subject(sentence)
            predicate = self.find_predicate(sentence)
            object_ = self.find_object(sentence)
            # print("triplet - ", subject, predicate, object_)
            return (subject, predicate, object_)
        except:
            return ()

    def treebank(self, sentence):
        tree = list(parser.raw_parse(sentence))[0]
        triple = self.main(ParentedTree.convert(tree))
        return triple

if __name__=="__main__" :
    import sys   
    # Parse the example sentence
    sent = "Astatine is a radioactive chemical element with the chemical symbol At and atomic number 85, and is the rarest naturally occurring element on the Earth's crust."
    t = TripleExtraction()
    t.treebank(sent)



