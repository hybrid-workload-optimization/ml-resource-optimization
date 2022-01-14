import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch

from ...utils import logger

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, name):
        super(CustomCallback, self).__init__()
        self.log = logger.get(self)
        self.name = name
        self.epoch = 0
        self.steps = 0
        self.total_steps = 0
                
    def update(self, epoch=None, batch=None, message=None):
        if epoch is not None:
            self.epoch = epoch

        if batch is not None:
            self.batch = batch
        
        try:
            #self.progress = int((self.epoch * self.steps + self.batch) / self.total_steps * 100)
            self.progress = (self.epoch * self.steps + self.batch) / self.total_steps * 100
        except:
            self.progress = 0
          
        notify = {
            "category":"model build",
            "step": self.name,
            "progress": self.progress,
            "message": message
        }
        # self.log.debug(f"status -- {notify}")

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']
        self.total_steps = self.epochs * self.steps
        self.update(epoch=0, batch=0, message="Starting training.")
        
    def on_train_end(self, logs=None):
        epochs = self.params['epochs']
        self.update(epoch=epochs, batch=0 , message="Training completed.")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.update(epoch=epoch, message="Working on traing.")        
        
    def on_train_batch_end(self, batch, logs=None):
        self.update(batch=batch, message="Working on traing.")
        