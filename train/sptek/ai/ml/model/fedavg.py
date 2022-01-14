import numpy as np
import tensorflow as tf

class FedAvg():
    
    def __init__(self, w):
        self._g_w_ = np.array(w, dtype=object)
    
    def process(self, w):
        w = np.array(w, dtype=object)
        for i, k in np.ndenumerate(w):
            self._g_w_[i] += k
        self._g_w_ = np.true_divide(self._g_w_, 2)
        
    def weights(self):
        return self._g_w_
    
    def compare(self, w):
        result = False
        w = np.array(w, dtype=object)
        for i, k in np.ndenumerate(w):
            comparison = self._g_w_[i] == k
            result += comparison.all()
        return result
        