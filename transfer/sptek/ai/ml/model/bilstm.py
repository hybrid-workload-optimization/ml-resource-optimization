import tensorflow as tf
import time

from . import algorithm
from ... import prefix
from ...utils import logger
from ...serving import ImportAdaptor
from ...serving import ModelServing


class BiLSTM(algorithm.ModelHandler):
    
    def __init__(self, units, inputs, outputs, in_features, out_features,
                 epochs=None, loss=None, optimizer=None, metrics=None, earlystopping=None):
        
        super().__init__(units, inputs, outputs, in_features, out_features,
                         epochs, loss, optimizer, metrics, earlystopping)
        
        self.log = logger.get(self)
        self._algorithm_ = 'BiLSTM'
        
    def __call__(self, *args, **kwds):
        self.sequential = BiLSTM.create(self.units, self.inputs, self.outputs, self.in_features, self.out_features)
        return self.sequential
        
    @classmethod
    def create(cls, units, inputs, outputs, in_features, out_features):
        
        sequential = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=False, activation="tanh"), input_shape=(inputs, in_features)),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(outputs * out_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([outputs, out_features])
        ])
        
        return sequential
    
    def build(self):
        self.__call__()
        self.compile(self.sequential)
    
    def load(self, path):        
        try:
            file = prefix.model.name(path)
            adaptor = ImportAdaptor('keras', ModelServing())
            self.sequential = adaptor.get(file)
            super().load(path)
        except Exception as e:
            self.log.warn(f"{e}")
            
    def is_transfer(self):
        if hasattr(self, '_is_transfer_'):
            return self._is_transfer_
        return False
    
    def is_continual(self):
        if hasattr(self, '_is_continual_'):
            return self._is_continual_
        return False