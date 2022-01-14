import tensorflow as tf
import time
import os

from tensorflow import keras
from tensorflow.keras import backend as K

from .callback import CustomCallback
from ... import master
from ... import prefix
from ...serving import ExportAdaptor
from ...serving import ModelServing
from ...utils import logger


hyper_map = {
    'Adam': tf.optimizers.Adam(),
    'MeanSquaredError': tf.losses.MeanSquaredError(),
    'MeanAbsoluteError': tf.metrics.MeanAbsoluteError(),
    'MeanAbsolutePercentageError': tf.keras.metrics.MeanAbsolutePercentageError()
}


def _get_loss_(option):
    try:    
        return hyper_map[option]
    except:
        return tf.losses.MeanSquaredError()

def _get_optimizer_(option):
    try:
        return hyper_map[option]
    except:
        return tf.optimizers.Adam()

def _get_metrics_(option):
    if not isinstance(option, list):
        option = [option]
    
    try:
        metric = []
        for key in option:
            metric.append(hyper_map[key])
        return metric
    except:
        return None

def _get_earlystopping_(option):
    return tf.keras.callbacks.EarlyStopping(monitor=option.monitor, patience=option.patience, mode=option.mode)


class Model():
    def __init__(self, units, inputs, outputs, in_features, out_features,
                epochs=None, loss=None, optimizer=None, metrics=None, earlystopping=None):
        
        self.log = logger.get(self)
        self.units = units
        self.inputs = inputs
        self.outputs = outputs
        self.in_features = in_features
        self.out_features = out_features
        self.earlystopping = earlystopping
        
        self.epochs = epochs
        if epochs is None:
            self.epochs = 10
            
        self.loss = loss
        if loss is None:
            self.loss = 'MeanSquaredError'
            
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = 'Adam'
        
        self.metrics = metrics
        if metrics is None:
            self.metrics = ['MeanAbsoluteError', 'MeanAbsolutePercentageError']

        self._path_ = None
        self._algorithm_ = None
        self._freeze_ = False

    def compile_fit(self, window, model=None, epochs=None):
        self.compile(model)
        self.fit(model, window, epochs)

    def compile(self, model):
        if model is not None:
            optimizer=_get_optimizer_(self.optimizer)
            
            if self._freeze_:
                lr = K.get_value(optimizer.learning_rate)
                self.log.info(f"learning_rate: {lr} --> {lr/10}")
                optimizer.learning_rate.assign(lr/10)
                
            model.compile(loss=_get_loss_(self.loss),
                                optimizer=optimizer,
                                metrics=_get_metrics_(self.metrics))

    def fit(self, model, window, epochs):
        callback = None
        if self.earlystopping is not None:
            callback = _get_earlystopping_(self.earlystopping)
            print(f"applay callback early_stopping...")
            
        if epochs is None:
            epochs = self.epochs
            
        model.fit(window.train,
                  epochs=epochs,
                  validation_data=window.val,
                  callbacks=[callback, CustomCallback(self.__class__.__name__)],
                  use_multiprocessing=True)
        
    def load(self, path):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
    
    def export(self, path, model=None):
        raise NotImplementedError
    
    def freeze(self, fine=0.9):
        self._freeze_ = True
        self._fine_ = fine

    def path(self):
        return self._path_
    
    def algorithm(self):
        return self._algorithm_


class ModelHandler(Model):
    
    def __init__(self, units, inputs, outputs, in_features, out_features,
                 epochs=None, loss=None, optimizer=None, metrics=None, earlystopping=None):
        
        super().__init__(units, inputs, outputs, in_features, out_features,
                         epochs, loss, optimizer, metrics, earlystopping)
        
        self._empty_ = True
        self._path_ = None
        self._completed_ = False
        
        
    def compile_fit(self, wndow=None, model=None):
        start = time.time()
        if model is None:
            model = self.sequential
        super().compile_fit(wndow, model)
        self._empty_ = False
        duration =  time.time() - start
        self.log.info(f"compile and fit machine learning model completed.-- (duration= {duration:.5f}s)")
        
    def compile(self, model=None):
        if model is None:
            model = self.sequential
        super().compile(model)
        self._empty_ = False
        
    def fit(self, window=None, model=None):
        start = time.time()
        if model is None:
            model = self.sequential            
        super().fit(model, window, None)
        duration =  time.time() - start
        self.log.info(f"fit machine learning model completed.-- (duration= {duration:.5f}s)")
        self._completed_ = True
        # self.window = window
        
    def load(self, path):
        self._path_ = path
        if self.sequential:
            self._empty_ = False

    def export(self, path, model=None):
        try:
            path = prefix.model.name(path)
            serving = ModelServing()
            adaptor = ExportAdaptor('keras', serving)
            adaptor.put({
                "model": self,
                "msg": self._algorithm_}, path)
        except Exception as e:
            self.log.warn(f"\n{e}")
    
    def predict(self, dataset):
        _output_ = self.sequential.predict(dataset)
        return _output_
    
    def save(self, path):
        self.sequential.save(path)
    
    def load_weights(self, path):
        self.sequential.load_weights(path)
    
    def get_weights(self):
        return self.sequential.get_weights()
    
    def set_weights(self, weights):
        self.sequential.set_weights(weights)
        
    def save_weights(self, path):
        self.sequential.save_weights(path)
        
    def freeze(self, fine=0.9):
        super().freeze(fine)
        self.sequential.trainable = True
        
        self.log.info(f"Number of layers in the base model: {len(self.sequential.layers)}")
        
        fine_tune_at = int(len(self.sequential.layers) * fine)
        if fine_tune_at < 1:
            raise Exception(f"Layers cannot be freeze. (freeze: {fine_tune_at})")
        
        for layer in self.sequential.layers[:fine_tune_at]:
            layer.trainable =  False
    
    def empty(self):
        return self._empty_
    
    def is_completed(self):
        return self._completed_