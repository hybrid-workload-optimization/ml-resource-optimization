import time
import math
import warnings
import numpy as np
import collections

import joblib
from pathlib import Path
# from joblib import dump, load
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from ...master import Master
from ...utils import logger, path as upath
from ...serving import ModelServing
from ... import prefix

def _calc_distance_(x1, y1, a, b, c):
    d = abs((a *x1 + b * y1 + c) / (math.sqrt(a * a + b * b)))
    return d

class KMeansCluster(object):
    
    def __init__(self, master, searcher, clusters=2, random_state=0, export_path=None, import_path=None):
        self.log = logger.get(self)
        self.master = master
        self.random_state = random_state
        self.clusters = clusters
        self._kmeans_list_ = []
        self._kmeans_current_ = None
        self.dataset(searcher=searcher)

    def _new_kmeans_(self, n_clusters):
        return MiniBatchKMeans(n_clusters = n_clusters, init="k-means++",random_state=0)

    def dataset(self, searcher):
        self._searcher_ = searcher
        
    def searcher(self):
        return self._searcher_

    def dense_vector(self):
        return self.searcher()['dense_vector']
    
    def squeeze(self):
        return self.dense_vector()
    
    def load(self, path):
        try:
            serving = ModelServing()
            self._kmeans_current_ = serving.get('kmeans', prefix.model.name(path))
            
            self._kmeans_list_.append(self._kmeans_current_)
            self.n_clusters = self._kmeans_current_.n_clusters
        except Exception as e:
            self.log.warn(f"{e}")

    def fit(self):
        start = time.time()
        
        self._dense_vector_ = self.squeeze()
        if self.clusters == "auto":
            self.n_clusters = self.elbow(self._dense_vector_, 10)
            self._kmeans_current_ = self._kmeans_list_[self.n_clusters-1]
        else:
            self.n_clusters = self.elbow(self._dense_vector_, self.clusters)
            self._kmeans_current_ = self._kmeans_list_[self.n_clusters-1]
        
        duration =  time.time() - start
        self.log.info(f"all subprocess for build kmeans cluster completed. -- {self.n_clusters} -- (duration={duration:.5f}s)")
                    
    def predict(self, vector):
        try:
            return self._kmeans_current_.predict(vector)
        except Exception as e:
            raise e

    def export(self, path):
        try:
            info = {
                "model": self._kmeans_current_,
                "msg": 'kmeans'
            }
            serving = ModelServing()
            serving.put('kmeans', info, prefix.model.name(path))
        except Exception as e:
            self.log.warn(f"{e}")

    def elbow(self, vector, clusters):
        if vector is None:
            self.log.warn(f"The _squeeze_vector_ variable value is invalid.")
            return
        
        _inertia_ = []
        _K_ = []

        for i in range(1, clusters+1):
            _K_.append(i)
            kmeans = self._new_kmeans_(i)
            kmeans.fit(vector)
            _inertia_.append(kmeans.inertia_)
            self._kmeans_list_.append(kmeans)
            self.log.info(f"clustering --  KMeans(n_clusters={kmeans.n_clusters}, random_state={kmeans.random_state})")
            self.log.debug(f"clustering -- inertia: {kmeans.inertia_} -- KMeans(n_clusters={kmeans.n_clusters}, random_state={kmeans.random_state}, n_iter={kmeans.n_iter_})")

        if logger.logVerbose == 0:
            self.debug_plot(
                title='The Elbow Method',
                x_label='Number of clusters',
                y_label='inertia',
                data=_inertia_,
                path='/root/dev/cmpai-v3-anomaly/debug/Elbow-Method.png'
            )

        _optimum_k_ = self._auto_detect_k_(_K_, _inertia_)
        return _optimum_k_

    def _auto_detect_k_(self, K, inertia):
        a = inertia[0] - inertia[-1]
        b = K[-1] - K[0]
        c1 = K[0] * inertia[-1]
        c2 = K[-1] * inertia[0]
        c = c1 - c2

        distance_of_point_from_line = []
        for i in range(len(K)):
            distance_of_point_from_line.append(
                _calc_distance_(K[i], inertia[i], a, b, c)
            )

        if logger.logVerbose == 0:
            self.debug_plot(
                title='The distance of Elbow Method',
                x_label='clusters',
                y_label='distance of point',
                data= distance_of_point_from_line,
                path='/root/dev/cmpai-v3-anomaly/debug/Elbow-Method-distance.png'
            )

        self.log.debug(f"clusters: {list(range(len(K)))}")
        self.log.debug(f"inertia: {inertia}")
        self.log.debug(f"distance: {distance_of_point_from_line}")

        _optimum_k_ = distance_of_point_from_line.index(max(distance_of_point_from_line))+1
        return _optimum_k_

    def debug_plot(self, title, x_label, y_label, data, path):
        plt.figure()
        plt.plot(range(0, len(data)), data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.savefig(path)

    def __repr__(self):
        labels = collections.Counter(self._kmeans_current_.labels_)
        head = [
            f"{super().__repr__()}",
            f"* category size : {self.n_clusters}"
        ]
        for i in range(self.n_clusters):
            head.append(f"* category.{i} size : {labels[i]}")
            
        return '\n'.join(head)