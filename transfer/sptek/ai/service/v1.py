import json
import time
from datetime import datetime

from flask import request
from flask import Response
from flask_restx import Namespace
from flask_restx import Resource

from .parser import RequestParser
from .. import prefix
from ..dataset import DataFactory
from ..dataset import DataSearcher
from ..dataset import SearcherFactory
from ..dataset import SplitterFactory
from ..generater import GeneratorFactory
from ..generater import TimestampFactory
from ..master import Master
from ..ml import ContinualModel
from ..ml import EWC
from ..ml import FedAvg
from ..ml import ModelFactory
from ..ml import ModelFeature
from ..outlier import OutlierFactory
from ..repository import ReposFactory
from ..repository import CSVAdapterBuilder
from ..repository import DatabaseAdaptorBuilder
from ..repository import ExportAdaptorBuilder
from ..repository import ImportAdaptorBuilder
from ..repository import KerasAdaptorBuilder
from ..repository import KerasAdapterWeightsBuilder
from ..repository import KmeansAdapterBuilder
from ..repository import ScalerAdapterBuilder
from ..scaler import ScalerFactory
from ..prefix import DataPath
from ..prefix import ModelPath
from ..prefix import ScalerPath
from ..prefix import TargetPath
from ..rest import rest
from ..scheduler import JobTrigger
from ..scheduler import SchedulerFactory
from ..serving import ExportAdaptor
from ..serving import ImportAdaptor
from ..serving import ModelServing
from ..utils import logger
from ..utils import serialize
from ..utils import stamp
from ..utils import MetaclassSingleton
from ..utils import ThreadPool


Transfer = Namespace("Transfer")


def service_model_entry():
    # list up of model
    master = Master()
    meta = master.target(dtype='resource')
    # logger.debug(f"{meta}")
    
    values = meta.value
    if not isinstance(meta.value, list):
        values = [meta.value]        
    return values


def setup(master):
    try:
        log = logger.get("Setup")
        
        # use route connecton True / False
        log.info(f"use route connecton: {master._use_route_}")
        if master._use_route_:
            peroid = 1
            models = None
            endpoint = master.endpoint()
            
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
    
            response = rest.request_post(url, data)
            response.raise_for_status()
            log.info(response.json())
            
        # check service type
        if not master.service_type() in ['transfer']:
            raise NotImplementedError(f'Not supported type. (type= {master.service_type()})')
    
        # setup model serving.
        log.info(f"setup model serving.")
        r2 = ReposFactory.create(master)
        importer = ImportAdaptorBuilder(master)
        importer.append(KerasAdaptorBuilder(master))
        importer.append(KerasAdapterWeightsBuilder(master))
        importer.append(KmeansAdapterBuilder(master))
        importer.append(ScalerAdapterBuilder(master))
        importer.build(r2)
        
        database = DatabaseAdaptorBuilder(master)
        database.append(CSVAdapterBuilder(master))
        database.build(r2)
        
        exporter = ExportAdaptorBuilder(master)
        exporter.append(KerasAdaptorBuilder(master))
        exporter.append(KerasAdapterWeightsBuilder(master))
        exporter.append(KmeansAdapterBuilder(master))
        exporter.append(ScalerAdapterBuilder(master))
        exporter.build(r2)
        
        serving = ModelServing()
        serving.attach('import', importer)
        serving.attach('import', database)
        serving.attach('export', exporter)
        serving.start()
        
        # setup job scheduler.
        log.info(f"setup job scheduler.")
        searcherJob = createScheduleJob(master)
        with SchedulerFactory.create(master, name='searcher') as scheduler:
            if not scheduler.enable():
                log.info(f"not use job scheduler.")
            else:
                scheduler.add_job(searcherJob)
                scheduler.start()

        # searcherJob = createScheduleJob(master)
        # searcherJob.job()
    
    except Exception as err:
        log.error(err)
        exit()


def createScheduleJob(master):
    return TransferTrainService()


@Transfer.route('/train')
class TransferTrain(Resource):

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
            self.log.debug(f"parser: {parser}")
            
            service = TransferTrainService()
            service.for_train(parser)
            return rest.response_ok()
        
        except Exception as e:
            self.log.warning(e)

        return rest.response_failed()


@Transfer.route('/train/status')
class TransferTrainStatus(Resource):

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

@Transfer.route('/forecast')
class TransferForecast(Resource):
    
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
            
            service = TransferForecastService()
            session = service.fork_predict(parser)
            service.wait_completed(session)
            return Response(service.response(), mimetype='application/json', status=200)
        
        except Exception as e:
            self.log.warning(e)

        return rest.response_failed()



class TransferTrainService(JobTrigger, metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.log = logger.get(self.__class__.__name__)
        self.session = []

    def for_train(self, parser):
        pool = ThreadPool()        
        sess = pool.add_task_nowait(self.train, parser)
        self.session.append(sess)
        return sess
    
    def job(self):
        self.log.info(f"start scheduling")        
        self.train(
            RequestParser({
                "model" : None,
                "period" : None,
                # "dateTime" : datetime.now()
                "dateTime" : "2021-12-13 09:00:00"
            })
        )

    def train(self, parser, session=None):        
        if parser is None:
            raise Warning(f"'parser' object is 'NonType'.")
    
        target_entry = []
        entry = parser.model()
        if entry:
            if isinstance(entry, list):
                target_entry.extend(entry)
            if isinstance(entry, str):
                target_entry.extend([entry])
        else:
            target_entry.extend(service_model_entry())
            
        model = self.pre_model()
        scaler = self.pre_scaler()
        
        for target in target_entry:
            parser["model"] = target       
            model, scaler = self.serve(parser, model, scaler)
    
    def pre_model(self):
        # check use pre-model
        master = Master()
        meta = master.import_model(dtype='sequential')
        if meta:
            # load last pre-model
            target = ModelPath(master)
            model = ModelFactory.create(master, action='transfer')
            model.load(target.path())
            self.log.info(f"load pre-model ==> '{target.path()}'")
            return model
        return None
    
    def pre_scaler(self):
        master = Master()
        return ScalerFactory.load(master)
    
    def serve(self, parser, model, scaler=None, session=None):
        self.log.info(f"start transfer learning.")
        start = time.time()
        
        if parser is None:
            raise Warning(f"'parser' object is 'NonType'.")
    
        # master config set
        master = Master()
        
        # load dataset for next train
        target = DataPath(master)
        dataset = DataFactory.load(target.path())
        # logger.debug(f"\n{dataset}")
                
        # apply filter for get only data from the forecast period
        searcher = SearcherFactory.create(master, dataset)
        dataset = searcher.filter(parser.model(), parser.dateTime())
        if dataset.empty():
            raise Warning(f"Empty DataFrame ...")
        logger.debug(f"\n{dataset}")
        
        # generate only weekday from the forecast period (preprocessing)
        generator = GeneratorFactory.create(master)
        dataset = generator.transform(dataset)
        # logger.debug(f"\n{dataset}")
        
        # remove outlier data from the forecast period (preprocessing)
        outlier = OutlierFactory.create(master)
        if outlier:
            dataset = outlier.transform(dataset)
            # logger.debug(f"\n{dataset}")
        
        # apply scale data from the forecast period (preprocessing)
        # if not scaler:
        #     scaler = ScalerFactory.create(master)
        #     if scaler:
        #         dataset = scaler.fit_transform(dataset)
        # else:
        #     scaler.transform(dataset)
        
        scaler = ScalerFactory.create(master)
        if scaler:
            dataset = scaler.fit_transform(dataset)
        
        # apply data sampling of next train
        trainset = SplitterFactory.create(master, dataset)
        trainset.fit()
        # logger.debug(f"\n{trainset.generator()}")
                
        # build sequentional dataset
        window = trainset.generator()
        
        # build train model
        train = model
        if not model or model.empty():
            model = ModelFactory.create(master, feature=ModelFeature(window), action='train')
            model.compile()
            self.log.info(f"compile new transfer model")
            model.window = window
        
        # build transfer model
        if model.is_transfer():
            model.freeze()
            model.compile()
    
        # build continual learning model
        if model.is_continual():
            if model.is_completed():
                
                # create EWC (Elastic weigths consolidation)
                logger.debug(f"before window id : {id(model.window)}")
                sample = model.window.train
                sample_size = len(model.window.train)
                ewc = EWC(model, sample, sample_size)
                
                # create new model
                model = ModelFactory.create(master, feature=ModelFeature(window), action='train')
                model.compile()
                
                # create new continual model
                logger.debug(f"after window id : {id(window)}")
                model = ContinualModel(model, ewc)
                model.compile()
                model.window = window
                
        # train model
        model.fit(window)
        
        # export transfer learning modle.
        traget = ModelPath(master)    
        model.export(traget.path())
        
        # export scaler instance
        target = ScalerPath(master, parser.model())
        scaler.export(target.path())
        
        duration =  time.time() - start
        self.log.info(f"transfer learning completed -- (duration= {duration:.5f}s)")
        
        return model, scaler
        
class TransferForecastService(JobTrigger, metaclass=MetaclassSingleton):
    
    def __init__(self):
        self.log = logger.get(self.__class__.__name__)
        self.session = []
        self._source_val_ = None
        self._predict_val_ = None
        self._db_type_ = None
    
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
        # set json body
        body = {
            "model" : self._db_type_
        }
                
        # Packaging structural response forecast values.
        # build resource values.
        for key, value in self._source_val_.meta.export.items():
            body[key] = list(self._source_val_[value])
            body[key].extend(list(self._predict_val_[value]))
        
        # debug resource values.
        report = ""
        for key in self._source_val_.meta.export.keys():
            report = report + f" -- {key}: {min(body[key])} ~ {max(body[key])}"
        report = f"-- size: {len(body[key])}" + report
        self.log.info(f"forecast completed {report}")
        
        return json.dumps(
            body,
            default=serialize.json_serialize,
            indent=4
            )

    def job(self):
        self.log.info(f"start scheduling")
        self.serve(None)

    def serve(self, parser, session=None):
        start = time.time()
        
        # set request database type
        self._db_type_ = parser.model()
        
        # master config set
        master = Master()
        
        # load dataset for next train
        target = DataPath(master)
        dataset = DataFactory.load(target.path())
                
        # apply filter for get only data from the forecast period
        searcher = SearcherFactory.create(master, dataset, action="forecast")
        dataset = searcher.filter(parser.model(), parser.dateTime())
        if dataset.empty():
            raise Warning(f"Empty DataFrame ...")
        logger.debug(f"\n{dataset}")
        
        # generate only weekday from the forecast period (preprocessing)
        generator = GeneratorFactory.create(master)
        dataset = generator.transform(dataset)
        
        # remove outlier data from the forecast period (preprocessing)
        outlier = OutlierFactory.create(master)
        if outlier:
            dataset = outlier.transform(dataset)
        
        # apply scale data from the forecast period (preprocessing)
        scaler = ScalerFactory.create(master)
        if scaler:            
            dataset = scaler.transform(dataset)
            
        # apply data sampling of next train
        testset = SplitterFactory.create(master, dataset, action='forecast')
        input_set = testset.generator()
        
        # load pre-model
        traget = ModelPath(master)
        model = ModelFactory.load(traget.path(), action='forecast')
        _val_ = model.predict(input_set)
        _predict_ = scaler.inverse_transform(_val_[0])
        
        # build input values.
        _source_ = scaler.inverse_transform(input_set[0])
        input = dataset.copy()
        input.reset_index()
        input.update(_source_, feature=testset.columns())
        input.to_percent()
        self._source_val_ = input
        # logger.debug(f"_source_val_:\n{self._source_val_}")
        
        # build output values.
        output = dataset.copy()
        output.reset_index()
        output.update(_predict_, feature=testset.columns())
        output.to_percent()
        # logger.debug(f"_predict_val_:\n{self._predict_val_}")
        
        # build time line.
        self._request_time_ = parser.dateTime()
        timestamp = TimestampFactory.create(master, self._request_time_)
        output.update(timestamp.generator(), feature=timestamp.columns())        
        self._predict_val_ = output
        # logger.debug(f"_request_time_:\n{self._request_time_}")
        
        duration =  time.time() - start
        self.log.info(f"transfer learning completed -- (duration= {duration:.5f}s)")
    