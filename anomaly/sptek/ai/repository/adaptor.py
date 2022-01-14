import time
import joblib
import pickle
import tensorflow as tf
from ..ml import ModelFactory
from ..ml.model.bilstm import BiLSTM
from ..utils import logger

class Serving(object):
    
    def get(self, data):
        raise NotImplementedError
    
    def put(self, data):
        raise NotImplementedError
    
    def pull(self, data):
        raise NotImplementedError
    
    def update(self, *args, **kwds):
        raise NotImplementedError


class Adaptor(Serving):
    def __init__(self, master, file, service):
        self.log = logger.get(self)
        self.master = master
        self._file_ = file
        self.service = service
        self.target = None
        
    def file(self):
        return self._file_
    
    def is_update(self):
        update, _, _ = self.status(self.file())
        return update
    
    def status(self, file=None):
        return self.service.get(file)
        
    def get(self, file=None):
        update, path, status = self.status(file)
        if update or self.target is None:
           self.target = self.instance(path)
           self.service.update_status(status)
        return self.target
        
    def put(self, obj, file=None):
        if file is None:
            file = self.file()
        self.service.put(self.dump, obj, file)


class KerasAdapter(Adaptor):
    
    def __init__(self, master, file, service):
        super().__init__(master, file, service)
        self.log = logger.get(self)
    
    def instance(self, path):
        try:
            start = time.time()
            obj = tf.keras.models.load_model(path)
            duration =  time.time() - start
            self.log.info(f"import sequential model ==> {path} -- (duration={duration:.5f}s)")
            return obj
        except Exception as e:
            msg = f"Load Model file does not exist at: {path}"
            self.log.error(f"{msg}\n{e}")
            raise Exception(msg)
        
    def dump(self, obj, path):
        start = time.time()
        obj['model'].save(path)
        duration =  time.time() - start
        self.log.info(f"export sequential model ==> {path} -- (duration={duration:.5f}s)")
        return obj['msg']
    
    def update(self, *args, **kwds):
        self.get(self.file())
        # self.log.info(f"update sequential model completed.")


class KerasAdapterWeights(KerasAdapter):
    
    def __init__(self, master, file, service):
        super().__init__(master, file, service)
        self.log = logger.get(self)
    
    def instance(self, path):
        try:
            start = time.time()            
            filename = path.joinpath(path.name)
            meta = self.model_from_json(filename.with_suffix(".json"))
            model = BiLSTM(meta['units'],
                           meta['inputs'],
                           meta['outputs'],
                           meta['in_features'],
                           meta['out_features'],
                           meta['epochs'],
                           meta['loss'],
                           meta['optimizer'],
                           meta['metrics'])
            model.build()
            model.load_weights(filename)
            
            duration =  time.time() - start
            self.log.info(f"import sequential model ==> {path} -- (duration={duration:.5f}s)")
            return model
        except Exception as e:
            msg = f"Load Model file does not exist at: {path}"
            self.log.error(f"{msg}\n{e}")
            raise Exception(msg)
      
    def dump(self, obj, path):
        start = time.time()
        path.mkdir(parents=True, exist_ok=True)
        filename = path.joinpath(path.name)
        obj['model'].save_weights(filename)
        self.model_to_json(obj['model'], filename.with_suffix(".json"))
        duration =  time.time() - start
        self.log.info(f"export sequential model ==> {path} -- (duration={duration:.5f}s)")
        return obj['msg']
    
    def model_to_json(self, model, path):
        meta = {
            'units': model.units,
            'inputs': model.inputs,
            'outputs': model.outputs,
            'in_features': model.in_features,
            'out_features': model.out_features,
            'epochs': model.epochs,
            'loss': model.loss,
            'optimizer': model.optimizer,
            'metrics': model.metrics
        }
        with open(path, 'wb') as fp:
            pickle.dump(meta, fp)
           
    def model_from_json(self, path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)
        

class KmeansAdapter(Adaptor):
    
    def __init__(self, master, file, service):
        super().__init__(master, file, service)
        self.log = logger.get(self)
        
    def instance(self, path):
        start = time.time()
        obj = joblib.load(path)
        duration =  time.time() - start
        self.log.info(f"import kmeans model ==> {path} -- (duration={duration:.5f}s)")
        return obj
        
    def dump(self, obj, path):
        start = time.time()
        joblib.dump(obj['model'], str(path))
        duration =  time.time() - start
        self.log.info(f"export kmeans model ==> {path} -- (duration={duration:.5f}s)")
        return obj['msg']

    def update(self, *args, **kwds):
        self.get(self.file())
        # self.log.info(f"update kmeans model completed.")


class ScalerAdapter(Adaptor):
    
    def __init__(self, master, file, service):
        super().__init__(master, file, service)
        self.log = logger.get(self)
        
    def instance(self, path):
        start = time.time()
        obj = joblib.load(path)
        duration =  time.time() - start
        self.log.info(f"import scaler model ==> {path} -- (duration={duration:.5f}s)")
        return obj
        
    def dump(self, obj, path):
        start = time.time()
        joblib.dump(obj['model'], str(path))
        duration =  time.time() - start
        self.log.info(f"export scaler model ==> {path} -- (duration={duration:.5f}s)")
        return obj['msg']

    def update(self, *args, **kwds):
        self.get(self.file())
        # self.log.info(f"update scaler model completed.")

            