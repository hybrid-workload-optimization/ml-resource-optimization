import os
import time
import tensorflow as tf
from tensorflow import keras

from .callback import CustomCallback
from ... import master
from ...utils import logger
from ...serving import ModelServing
from ... import prefix


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

    def compile_fit(self, window, model=None, epochs=None):
        self._compile_(model)
        self._fit_(model, window, epochs)

    def _compile_(self, model):
        if model is not None:
            model.compile(loss=_get_loss_(self.loss),
                                optimizer=_get_optimizer_(self.optimizer),
                                metrics=_get_metrics_(self.metrics))

    def _fit_(self, model, window, epochs):
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
    
    def export(self, model, path):
        raise NotImplementedError

    def path(self):
        return self._path_
    
    def algorithm(self):
        return self._algorithm_


class ModelHandler(Model):
    
    def __init__(self, units, inputs, outputs, in_features, out_features,
                 epochs=None, loss=None, optimizer=None, metrics=None, earlystopping=None):
        
        super().__init__(units, inputs, outputs, in_features, out_features,
                         epochs, loss, optimizer, metrics, earlystopping)
        
        
    def compile_fit(self, wndow=None, model=None):
        start = time.time()
        if model is None:
            model = self.sequential
        super().compile_fit(wndow, model)
        duration =  time.time() - start        
        self.log.info(f"compile and fit machine learning model completed.-- (duration={duration:.5f}s)")

    def export(self, path, model=None):
        if model is None:
            model = self.sequential
        
        try:
            info = {
                "model": self,
                "msg": self._algorithm_
            }
            serving = ModelServing()
            serving.put('keras', info, prefix.model.name(path))
            
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