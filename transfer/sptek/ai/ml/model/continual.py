import tensorflow as tf
from tensorflow import keras
from . import algorithm
from .callback import CustomCallback
from ... import prefix
from ...serving import ExportAdaptor
from ...serving import ModelServing


class ContinualModel(keras.Model):
    
    def __init__(self, model, ewc, lambda_=0.1):
        super(ContinualModel, self).__init__()
        self.copy_parameter(model)
        self._target = model
        self.sequential = model.sequential
        self.prior_weights = self.sequential.get_weights()
        self.lambda_ = lambda_
        self.ewc = ewc
        # self.loss_tracker = keras.metrics.Mean(name="loss")
        
    def compile(self):
        super(ContinualModel, self).compile(
            loss=algorithm._get_loss_(self._m_loss),
            optimizer=algorithm._get_optimizer_(self._m_optimizer),
            metrics=algorithm._get_metrics_(self._m_metrics)
        )
        
        # get ewc (elastic weigth  consolidation)
        self.fisher_matrix = self.ewc.get_fisher()

    def train_step(self, data):
        
        # self.compiled_metrics.reset_state()
        # self.compiled_loss.reset_states()
        
        model = self.sequential
        # self.loss_tracker.reset_state()
        
        # Unpack the data.
        x, y = data
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)  # Forward pass
            
            # Compute our own loss
            # loss = keras.losses.mean_squared_error(y, y_pred)
            # loss = model.loss(y, y_pred)
            # loss = model.compiled_loss(y, y_pred)
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
            
            if self.fisher_matrix is not None:
                loss += self.compute_penalty_loss(model, self.fisher_matrix)
        
        # Compute gradients
        trainable_vars = model.trainable_variables
        gradients  = tape.gradient(loss, trainable_vars)
        
        # Update weights
        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
        # Update metrics (includes the metric that tracks the loss)
        # model.compiled_metrics.update_state(y, y_pred)
        # self.loss_tracker.update_state(loss)
        # self.compiled_metrics.update_state(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Update model
        self.sequential = model
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        # return {m.name: m.result() for m in model.metrics}
        # step_metric = {"loss": self.loss_tracker.result()}
        # step_metric.update({m.name: m.result() for m in model.metrics})
        # return step_metric
    
    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     # return [self.loss_tracker, self.metrics]
    #     return [self.metrics]
        

    def fit(self, window):
        super(ContinualModel, self).fit(window.train,
                  epochs=self._m_epochs,
                  validation_data=window.val,
                  use_multiprocessing=True)

    def compute_penalty_loss(self, model, fisher_matrix):
        penalty = 0.
        for u, v, w in zip(fisher_matrix, model.weights, self.prior_weights):
            penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
        return 0.5 * self.lambda_ * penalty
    
    def copy_parameter(self, model):
        self._m_path = None
        self._m_completed = False
        self._m_empty = model._empty_
        self._m_algorithm = model._algorithm_
        self._m_freeze = model._freeze_
        self._m_units = model.units
        self._m_inputs = model.inputs
        self._m_outputs = model.outputs
        self._m_in_features = model.in_features
        self._m_out_features = model.out_features
        self._m_earlystopping = model.earlystopping
        self._m_epochs = model.epochs
        self._m_loss = model.loss
        self._m_optimizer = model.optimizer
        self._m_metrics = model.metrics
        
    def empty(self):
        return False
    
    def is_transfer(self):
        return False
    
    def is_continual(self):
        return True
    
    def is_completed(self):
        return True
    
    def export(self, path, model=None):
        try:
            path = prefix.model.name(path)
            serving = ModelServing()
            adaptor = ExportAdaptor('keras', serving)
            adaptor.put({
                "model": self._target,
                "msg": self._target._algorithm_}, path)
        except Exception as e:
            self.log.warn(f"\n{e}")
    

