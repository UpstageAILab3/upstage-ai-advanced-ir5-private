import numpy as np

class SortVotingOneDocument:

    def __init__(self, documents, size=5 ):
        self.score_list = np.array([doc[0] for doc in documents])
        self.size = size

    def get_score_list(self):
        return self.score_list
    
    def softmax(self):
        x = self.score_list
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def probability(self):
        return self.softmax()
    
    def selected_indice(self):
        return np.argsort(self.probability())

# class SortVotingMultiDocument:
#     def __init__(self, **args):
#         first_scores = 