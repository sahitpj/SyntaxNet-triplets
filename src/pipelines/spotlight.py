import sys
sys.path.append("../../../..")
sys.path.append("../..")

from GSoC2019.pyspotlight import spotlight



class Spotlight_Pipeline(object):

    def __init__(self):
        self.spotlight_config = spotlight.Config()
        self.spotlight_address = self.spotlight_config.spotlight_address

    def read_annotations(self, annotations):
        return [ i['URI'] for i in annotations ]

    def annotate_word(self, word):
        try:
            annotations = spotlight.annotate(self.spotlight_address,
                                        word)
            return self.read_annotations(annotations)
        except spotlight.SpotlightException:
            print("URI not found")
            return word



    
        

    


