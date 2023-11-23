import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch


class Checkpoint(object):
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.save_weights_only = True
        self.verbose = 1
        
        self.callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                    save_weights_only=self.save_weights_only,
                                                    verbose=self.verbose)

    def latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.checkpoint_path.parent)

    def __repr__(self):
        return '\n'.join([
            f'{self.callback}',
            f'checkpoint path: {self.checkpoint_path.parent}',
            f'save weights only: {self.save_weights_only}',
            f'verbose: {self.verbose}'])



class RecoveryCallback(tf.keras.callbacks.Callback):

    def __init__(self, recovery, metrics="val_accuracy"):
        super(RecoveryCallback, self).__init__()
        self.recovery = recovery
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs=None):        
        epoch = epoch + 1
        marker = {"epochs":epoch}
        for key in self.metrics:
            marker[key] = logs[key]
        self.recovery.marking(marker)
        self.recovery.save()
