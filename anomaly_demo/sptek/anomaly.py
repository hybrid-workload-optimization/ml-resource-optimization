import json
import string
from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np
import rrcf
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import ai.gpus as gpus
from ai import service
from ai.master import Master
from ai.utils import logger, serialize
from ai.searcher import SearcherFactory
from ai.model import ModelFactory
from ai.dataset import CMPDataSet
from ai.rest import RESTful


# require tesorflow hub model
#  --> word2vec-wiki-words-250-with-normalization
#  --> word2vec-wiki-words-500-with-normalization

def execute(executer):
    logger.debug(f"executer: {executer}")
    logger.debug(f"executer: {executer.master}")

    if executer.command != "anomaly":
        return

    logger.file(executer.log)
    logger.verbose(executer.verbose)

    # test_execute(executer)

    # load master file
    master = Master(executer.master)
    master.load()
    master.temp_dir = str(Path(__file__).parent.parent.absolute()) + '/temp'

    # setup service
    service.v1.setup(master)
    
    # RESTful API 서비스 실행
    rest = RESTful()
    rest.api.add_namespace(service.v1.Forecast, '/v1')
    rest.start(master.port())
    rest.join()



def test_execute(executer):
    # initailize gpu
    gpus.set_device_configuration(size=executer.worker*320)

    # load master file
    master = Master(executer.master)
    master.load()
    master.temp_dir = str(Path(__file__).parent.parent.absolute()) + '/temp'

    # create searcher
    searcher = SearcherFactory.create(master)
    searcher.search("2021-09-15 00:00:00", "2021-09-15 00:10:00")
    searcher.log.info(f"collect logs is completed. ({searcher.count()})")
    test_rcf(searcher, master)
    
    # searcher.search("2021-09-15 00:00:00", "2021-09-15 00:30:00")
    # searcher.log.info(f"collect logs is completed. ({searcher.count()})")
    # test_cluster(searcher, master) 
    





def test_cluster(searcher, master):
    # load dataset
    ds = CMPDataSet(searcher)
    ds.build()
    # print(ds._dataset_)

    # words to vector
    wv = ModelFactory.create(master)
    wv.load()
    ds = wv.embedding(ds)    
    # print(ds)

    kmeans = KMeans(n_clusters=2, random_state=0) 
    kmeans.fit(ds)
    print(f"inertia={kmeans.inertia_}")
    # print(f"labels={kmeans.labels_}")
        
    max_clusters = 5
    cs = []
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(ds)
        cs.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters), cs)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('CS')
    plt.show()
    plt.savefig('fig-cluster.png')




def test_rcf(searcher, master):

    # load dataset
    ds = CMPDataSet(searcher)
    ds.build()
    ds.set_index()
    ds.drop_duplicates()
    ds.sort()
    print("load dataset completed.")
    
    # Create events
    events = {
    'warnning'  : ('2021-09-15 00:04:00',
                   '2021-09-15 00:04:10'),
    'error'     : ('2021-09-15 00:05:30',
                   '2021-09-15 00:06:00')
    }

    # generate test data
    for event, duration in events.items():
        start, end = duration
        message = ds._dataset_.loc[start, 'message']

        if event == "warnning":            
            ds._dataset_.loc[start:end, 'message'] = message + ' warning'
        
        if event == "error":
            ds._dataset_.loc[start:end, 'message'] = message + ' error'

    #print(ds._dataset_)
    print("creaet events completed.")

    # words to vector
    wv = ModelFactory.create(master)
    wv.load()
    ds = wv.embedding(ds)    
    # print(ds)
    print("words to embedding vector completed.")


    # Generate data
    X = ds
    n = X.shape[0]
    x = np.arange(n)

    # Set tree parameters
    num_trees = 100
    tree_size = 256

   # Construct forest
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly from point set
        ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                            replace=False)
        # Add sampled trees to forest
        trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
        forest.extend(trees)

    # Compute average CoDisp
    avg_codisp = pd.Series(0.0, index=np.arange(n))
    index = np.zeros(n)
    for tree in forest:
        codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index
    
    
    avg_codisp = ((avg_codisp - avg_codisp.min())
             / (avg_codisp.max() - avg_codisp.min()))
    
    fig, ax = plt.subplots(2, figsize=(10, 6))

    embedding_sum = pd.Series(X.sum(axis=1))
    embedding_sum.plot(ax=ax[0], color='0.5', alpha=0.8, label='Embedding vector summation')

    avg_codisp.plot(ax=ax[1], color='#E8685D', alpha=0.8, label='Random Cut Forest (RRCF)')

    ax[0].legend(frameon=True, loc=2, fontsize=10)
    ax[1].legend(frameon=True, loc=2, fontsize=10)

    ax[0].set_xlabel('')
    ax[1].set_xlabel('')

    ax[0].set_ylabel('word embedding vector summation', size=10)
    ax[1].set_ylabel('Normalized Anomaly Score', size=10)
    ax[0].set_title('Anomaly detection on CMP logs', size=14)
    ax[0].xaxis.set_ticklabels([])

    shift = 0.15
    ax[0].set_ylim(embedding_sum.min() - shift, embedding_sum.max() + shift)
    ax[1].set_ylim(avg_codisp.min() - shift, avg_codisp.max() + shift)
    plt.tight_layout()
    plt.show()
    plt.savefig('fig1.png')

    print("svae fig completed.")
