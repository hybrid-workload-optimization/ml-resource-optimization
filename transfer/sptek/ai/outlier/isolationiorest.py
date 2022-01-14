

class Outlier(object):
    
    def transform(self, dataset):
        raise NotImplementedError


class IsolationForest(Outlier):
    
    def transform(self, dataset):
        pass