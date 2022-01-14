import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, Response
from flask_restx import Namespace, Resource


from .parser import RequestParser
from .. import workspace as ws, prefix
from ..utils import logger, MetaclassSingleton, ThreadPool, stamp, serialize
from ..rest import rest
from ..searcher import SearcherFactory
from ..master import Master
from ..dataset import DatasetFactory, DataSearcher
from ..preprocessor import PreprocessorFactory, ScalerFactory
from ..ml import ModelFactory, ClusterFactory, ModelFeature
from ..bucket import BucketFactory, BucketSearcher
from ..repository import ReposFactory
from ..utils import encrypt
from ..prefix import ModelPath, ClusterPath, ScalerPath
from ..scheduler import JobTrigger, SchedulerFactory
from ..serving import ModelServing, DataAdaptorFactory, DataServing
from ..anomalize import AnomalyFactory


Train = Namespace("Train")
Forecast = Namespace("Forecast")

sin = None

def setup(master):    
    log = logger.get("Setup")

    log.info(f"use route connecton: {master._use_route_}")
    if master._use_route_:
        # registry anomaly info to router
        peroid = 1
        endpoint = master.endpoint()
        models = ws.model_entry(master)
        
        data = {
            "model": models,
            "period": peroid,
            "path": endpoint,
            "ip": master.ip(),
            "port": master.port()
        }

        url = f"http://{master.route()}/v1/registry"
        
        log.debug(f"data ==> {data}")
        log.debug(f"url ==> {url}")
    
        try:
            response = rest.request_post(url, data)
            response.raise_for_status()
            log.info(response.json())
        except Exception as err:
            log.error(err)
            
            
    try:
        # ml model continer and load
        log.info(f"setup model serving.")
        repos = ReposFactory.create(master)
        serving = ModelServing()
        serving.attach(repos)
        serving.start()

        # Create DataServing.
        log.info(f"setup job scheduler.")
        scheduler = SchedulerFactory.create(master, name='searcher')
        dtype, adaptors = DataAdaptorFactory.create(master)
        for a in adaptors:
            a.dateTime = "2021-11-28 03:00:00"
            a.connect()
            
        serving = DataServing()
        serving.attach(dtype, adaptors)
        serving.attach('scheduler', scheduler)
        # serving.start()
        pass
        
    except Exception as e:
        log.error(e)

def createScheduleJob(master):
    meta = master.database()
    # logger.debug(meta)
    
    parser = RequestParser({
        "model" : 'HW-Log',
        "period" : None,
        # "dateTime" : datetime.now()
        "dateTime" : "2021-11-28 03:00:00"
    })
    
    # create data searcher
    searcher = SearcherFactory.create(master, parser=parser, dtype='train')
    searcher.fork()
    searcher.wait_completed()
    
    # create train service
    trainService = AnomalyTrainService()
    trainService.parser = parser
    
    # create forcast service
    forecastService = AnomalyForecastService()
    forecastService.parser = parser
    
    return searcher, trainService, forecastService


@Train.route('/train')
class AnomalyTrain(Resource):

    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, args, kwargs)
        self.log = logger.get(self.__class__.__name__)

    def post(self):
        try:
            self.log.info(f"{request.path}")
            
            master = Master()
            endpoints = master.endpoint()
            if not request.path in endpoints:
                return rest.response_failed()
            
            parser = RequestParser(request.get_json(silent=True))
            # self.log.debug(f"parser: {parser}")
            service = AnomalyTrainService()
            service.start(parser)
            return rest.response_ok()
        
        except Exception as e:
            self.log.warning(e)

        return rest.response_failed()


@Train.route('/train/status')
class AnomalyTrainStatus(Resource):

    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, args, kwargs)
        self.log = logger.get(self.__class__.__name__)

    def post(self):
        try:
            self.log.info(f"{request.path}")
            return rest.response_ok()
        except Exception as e:
            self.log.warning(e)

        return rest.response_failed()


class AnomalyTrainService(JobTrigger, metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.log = logger.get(self.__class__.__name__)
        self.forecast = None
        self.session = []

    def start(self, parser):
        pool = ThreadPool()
        sess = pool.add_task_nowait(self.serve, parser)
        self.session.append(sess)
        
    def job(self):
        self.log.info(f"start scheduling -- train anomaly model.")
        self.serve(self.parser)

    def serve(self, parser, session=None):        
        start = time.time()       
        
        # master config set
        master = Master()
        
        # ds = DataServing()
        ds = DataServing()
        adaptor = ds.get('train', parser.model())
        adaptor.job(dateTime = parser.dateTime())
        dataset = adaptor.dataset
        
        # clustering logs    
        cluster_path  = ClusterPath(master)
        cluster = ClusterFactory.create(master, DataSearcher(dataset))
        cluster.fit()
        cluster.export(cluster_path.path())
        self.n_clusters = cluster.n_clusters
        
        # create term record
        bucket = BucketFactory.create(master, cluster=cluster, searcher=DataSearcher(dataset))
        bucket.fork_fit(1)
        bucket.wait_completed()
        
        # train dataset
        trainset = DatasetFactory.create(master, searcher=BucketSearcher(bucket), action="sampling")
        trainset.fit()
        
        # trainset data range to z-score (nomalize)
        scaler_path  = ScalerPath(master)
        scaler = ScalerFactory.create(master)
        scaler.fit_transform(trainset)
        scaler.export(scaler_path.path())
        
        # build LSTM model
        window = trainset.generator()
        model_path = ModelPath(master, parser)
        model = ModelFactory.create(master, feature=ModelFeature(window), action='train')
        model.compile_fit(window)
        model.export(model_path.path())
        
        self.forecast = model
        
        duration =  time.time() - start
        self.log.info(f"train anomaly model completed. -- (duration={duration:.5f}s)")


@Forecast.route('/forecast')
class AnomalyForecast(Resource):
    
    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, args, kwargs)
        self.log = logger.get(self.__class__.__name__)

    def post(self):
        try:
            self.log.info(f"{request.path}")
            
            master = Master()
            endpoints = master.endpoint()
            if not request.path in endpoints:
                return rest.response_failed()
            
            parser = RequestParser(request.get_json(silent=True))
            # self.log.debug(f"parser: {parser}")
            service = AnomalyForecastService()
            session = service.fork_predict(parser)
            service.wait_completed(session)
            return Response(service.response(), mimetype='application/json', status=200)
        
        except Exception as e:
            self.log.warning(e)

        return rest.response_failed()


class AnomalyForecastService(JobTrigger, metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.log = logger.get(self.__class__.__name__)
        self.session = []
        self.forecast = None
        self._predict_val_ = None
        self._db_type_ = None
        self._n_clusters_ = None
        
    def fork_predict(self, parser):
        pool = ThreadPool()
        sess = pool.add_task(self.serve, parser)
        self.session.append(sess)
        return sess
    
    def wait_completed(self, session):
        pool = ThreadPool()
        pool.wait_completion(session)
        self.session.remove(session)
        
    def response(self):
        # master config set
        master = Master()
        scaler_path  = ScalerPath(master)
        scaler = ScalerFactory.create(master)
        scaler.load(scaler_path.path())
        
        body = {
            "model" : self._db_type_
        }
        
        # Packaging structural response source values.
        # source value sahpe. (1, d1, d2) ---- d1 x d2
        category = scaler.inverse_transform(self._source_val_[0])
        for i in range(0, self._n_clusters_):
            label = f"category.{i}"
            body[label] = np.around(category[:,i].flatten())
        
        # Packaging structural response prediction values.
        # predict value sahpe. (1, d1, d2) ---- d1 x d2
        category = scaler.inverse_transform(self._predict_val_[0])
        for i in range(0, self._n_clusters_):
            label = f"forecast.{i}"
            body[label] = np.around(category[:,i].flatten())
            
        # Packaging structural response anomaly values.
        for i in range(0, self._n_clusters_):
            label = f"anomaly.{i}"
            anomlay = self._anomalize_val_[i]
            body[label] = np.around(list(anomlay))
        
        # Packaging structural response anomaly values.
        label = f"time"
        t, n = stamp._tick_and_nubmer_(self._interval_)
        tg = stamp.TimeGenerator.build(self._start_, self._end_, t, n)
        body[label] = list(tg)
            
        return json.dumps(
            body,
            default=serialize.json_serialize,
            indent=4
            )
    
    def job(self):
        self.log.info(f"start scheduling -- forecast anomaly model.")
        self.serve(self.parser)

    def serve(self, parser, session=None):
        start = time.time()
        
        self._db_type_ = parser.model()
        
        # master config set
        master = Master()
        
        ds = DataServing()
        adaptor = ds.get('forecast', parser.model())
        adaptor.job(dateTime = parser.dateTime())
        bucket = adaptor.bucket
        self._n_clusters_ = adaptor.cluster.n_clusters
        
        # to test dataset
        testset = DatasetFactory.create(master, searcher=BucketSearcher(bucket), action="forecast")
        testset.fit()
        
        # trainset data range to z-score (nomalize)
        scaler_path  = ScalerPath(master)
        scaler = ScalerFactory.create(master)
        scaler.load(scaler_path.path())
        scaler.transform(testset)
        
        # Load forecast model
        window = testset.generator()
        model_path = ModelPath(master, parser)
        model = ModelFactory.create(master, feature=ModelFeature(window), action='forecast')
        model.load(model_path.path())
        
        # forecast
        sub_start = time.time()
        input_set = window.test.take(1)
        self._source_val_ = window.get_input_val(input_set)
        self._predict_val_ = model.predict(input_set)
        duration =  time.time() - sub_start
        self.log.info(f"forecast trend of log completed. -- (duration={duration:.5f}s)")
        
        detector = AnomalyFactory.create(master)
        detector.fork_fit(self._predict_val_)
        detector.wait_completed()
        self._anomalize_val_ = detector.score()        
                
        self._end_ = bucket.current()
        self._start_ = stamp.prev_time(bucket.current(), bucket.ago())
        self._interval_ = bucket.interval()
        
        duration =  time.time() - start
        self.log.info(f"anomaly forecast completed. -- (duration={duration:.5f}s)")
       