import sys
sys.path.append("../../..")

from GSoC2019.conllu.conllu import parse_single, TokenList

# data_file = open("sample.conll", "r", encoding="utf-8")
# tokenlist = parse_single(data_file) #tokenlist gives the parsed conllu file
# print(tokenlist[0].to_tree())


class TripleExtraction(object):
    def __init__(self, filepath_to_conll):
        self.filepath_to_conll = filepath_to_conll
        data_file = open(filepath_to_conll, "r", encoding="utf-8")
        self.tokenlist = parse_single(data_file)
        self.tokenTree = tokenlist[0].to_tree()


    def treebank(self):
        





