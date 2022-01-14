import time
import math
import rrcf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import scipy.stats as stats

from ..utils import ThreadPool, logger, thread

def percentage_of_area_under_std_normal_curve_from_zcore(z_score):
    return .5 * (math.erf(z_score / 2 ** .5) + 1)

class AnomalyDetector(object):
    
    def __init__(self, master, num_trees=40, shingle_size=4, tree_size=256):
        self.log = logger.get(self)
        self.master = master
        
        # Set tree parameters
        self.num_trees = num_trees
        self.shingle_size = shingle_size
        self.tree_size = tree_size
        
        self.session = []
        self._session_completed_ = False

    def _wait_completed_(self):
        pool = ThreadPool()
        for s in self.session:
            pool.wait_completion(s)
        self.session = []

    def wait_completed(self):
        while not self._session_completed_:
            thread.nsleep(1)

    def compile_fit(self, data):
        forest, points, avg_codisp = self.compile(data)
        return self.fit(forest, points, avg_codisp)
        
    def fork_fit(self, data):
        pool = ThreadPool()
        self._score_ = {}
        for i in range(data.shape[len(data.shape)-1]):
            self._score_[i] = None
            self.session.append(pool.add_task(self._fork_serve_, data=data, i=i))
        pool.add_task(self._check_fork_)
            
    def _fork_serve_(self, data, i, session=None):
        self._score_[i] = self.compile_fit(data[:,:,i].flatten())
        
    def _check_fork_(self, session=None):
        start = time.time()
        self._wait_completed_()
        duration =  time.time() - start
        self.log.info(f"all subprocess for anomalize data completed. -- (0, 0) -- (duration={duration:.5f}s)")
        self._session_completed_ = True
    
    def score(self):
        dump = []
        for item in self._score_.values():
            dump.append(list(item))
        return dump
            
    def compile(self, data):
        raise NotImplementedError
        
    def fit(self):
        raise NotImplementedError



class BatchDetector(AnomalyDetector):
    
    def __init__(self, master, num_trees=40, shingle_size=4, tree_size=256):
        super().__init__(master, num_trees, shingle_size, tree_size)
        self.log = logger.get(self)        
        
    def compile(self, data):
        self.size = len(data)
        
        # Set forest parameters
        num_trees = 100
        tree_size = 256
        sample_size_range = (self.size // tree_size, tree_size)

        # Construct forest
        forest = []
        while len(forest) < num_trees:
            # Select random subsets of points uniformly
            ixs = np.random.choice(self.size, size=sample_size_range,
                                replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(data[ix], index_labels=ix)
                    for ix in ixs]
            forest.extend(trees)
            
    def fit(self):
        self.forest = []
        
        # Compute average CoDisp
        self.avg_codisp = pd.Series(0.0, index=np.arange(self.size))
        index = np.zeros(self.size)
        for tree in self.forest:
            codisp = pd.Series({leaf : tree.codisp(leaf)
                            for leaf in tree.leaves})
            self.avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        self.avg_codisp /= index
        self.threshold = self.avg_codisp.nlargest(n=10).min()
        
    def plot(self):
        fig = plt.figure(figsize=(12,4.5))
        ax = fig.add_subplot(121, projection='3d')
        sc = ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2],
                        c=np.log(self.avg_codisp.sort_index().values),
                        cmap='gnuplot2')
        plt.title('log(CoDisp)')
        ax = fig.add_subplot(122, projection='3d')
        sc = ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2],
                        linewidths=0.1, edgecolors='k',
                        c=(self.avg_codisp >= self.threshold).astype(float),
                        cmap='cool')
        plt.title('CoDisp above 99.5th percentile')



class StreamDetector(AnomalyDetector):
    
    def __init__(self, master, num_trees=40, shingle_size=4, tree_size=256):
        super().__init__(master, num_trees, shingle_size, tree_size)
        self.log = logger.get(self)
        
    def compile(self, data):
        n = len(data)
        # Create a forest of empty trees
        forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)
            
        # Use the "shingle" generator to create rolling window
        points = rrcf.shingle(data, size=self.shingle_size)

        # Create a dict to store anomaly score of each point
        # self.avg_codisp = {}
        avg_codisp = pd.Series(0.0, index=np.arange(n)).astype("float64")
        # self.data = data
        return forest, points, avg_codisp
    
    def fit(self, forest, points, avg_codisp):
        # For each shingle...
        for index, point in enumerate(points):
            # For each tree in the forest...
            for tree in forest:
                # If tree is above permitted size...
                if len(tree.leaves) > self.tree_size:
                    # Drop the oldest point (FIFO)
                    tree.forget_point(index - self.tree_size)
                # Insert the new point into the tree
                tree.insert_point(point, index=index)
                # Compute codisp on the new point...
                new_codisp = tree.codisp(index)
                # And take the average over all trees
                if not index in avg_codisp:
                    avg_codisp[index] = 0
                avg_codisp[index] += new_codisp / self.num_trees
        return avg_codisp
            
    def plot(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = 'tab:red'
        ax1.set_ylabel('Data', color=color, size=14)
        ax1.plot(self.data, color=color)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
        ax1.set_ylim(0,160)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('CoDisp', color=color, size=14)
        ax2.plot(pd.Series(self.avg_codisp).sort_index(), color=color)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
        ax2.grid('off')
        ax2.set_ylim(0, 160)
        plt.title('Sine wave with injected anomaly (red) and anomaly score (blue)', size=14)
        plt.show()
        plt.savefig('fig-stream_detector.png')