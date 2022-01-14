import time
import tensorflow as tf

from . import algorithm
from ...utils import logger
from ...serving import ModelServing
from ... import prefix

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
        self._compile_(self.sequential)
    
    def load(self, path):        
        try:
            serving = ModelServing()
            self.sequential = serving.get('keras', prefix.model.name(path))            
            if isinstance(self.sequential, dict):
                logger.debug(f"dict")
            else:
                logger.debug(f"model")
        except Exception as e:
            self.log.warn(f"{e}")
        