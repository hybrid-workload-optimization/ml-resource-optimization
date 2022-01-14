import numpy as np
import tensorflow as tf
from ..utils import path
from scipy.spatial import distance
# from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingVector(object):
    
    def __init__(self, meta):
        self.meta = meta
        self.path = path.uri_to_absolutized(meta.uri)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)

    def embedding(self, ds):
        col = []
        v = None
        #clone = ds.copy()
        for record in ds.messages():
            f = self.model([record])
            # print(f)
            #v = np.mean(f)
            # v = np.sum(f)
            # print(f"y = {v}") 
            col.append(f)
        # clone._dataset_['embed'] = col
        # return clone
        return np.vstack(col)
