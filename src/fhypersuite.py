import sys
sys.path.append("../../..")
sys.path.append("..")

from GSoC2019.hypernymysuite.hypernymysuite.base import HypernymySuiteModel

class FBHypernymBench(HypernymySuiteModel):

    HYPERNYMY_PREDICATES = ['is', 'was']

    def __init__(self):
        self.triplets_list = None
        self.hypernyms = None
        self.clean_hypernyms = None
        super().__init__(self)

    def get_hypernyms(self, triplets):
        '''
        triplets have the following form, (subject [attrs], predicate [attrs], object [attrs])
        In order to identify hypernyms, we find triplets which have a predicate of a helping word
        '''
        self.triplets_list = triplets
        hypernyms = list()
        clean_hypernyms = list()
        for triplet in triplets:
            predicate = triplet[1]
            if predicate[0] in HYPERNYMY_PREDICATES:
                hypernym = (triplet[0], triplet[2])
                clean_hypernym = (triplet[0][0], triplet[2][0])
                hypernyms.append(hypernym)
                clean_hypernyms.append(clean_hypernym)
        self.hypernyms = hypernyms
        self.clean_hypernyms = clean_hypernyms

    def predict(self, hypo, hyper):
        if (hypo, hyper) in self.clean_hypernyms or (hyper, hypo) in clean_hypernyms:
            return 1.
        else:
            return 0.



