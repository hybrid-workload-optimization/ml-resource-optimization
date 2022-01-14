import numpy as np
import tensorflow as tf
import tqdm
from .continual import ContinualModel


# class EWC(object):
    
#     def __init__(self, prior_model, data_samples, num_sample=30):
#         # self.prior_model = prior_model
#         # self.prior_weights = prior_model.get_weights()
        
#         if isinstance(prior_model, ContinualModel):
#             self.prior_model = prior_model.model
#             self.prior_weights = prior_model.model.get_weights()
#         else:
#             self.prior_model = prior_model
#             self.prior_weights = prior_model.get_weights()
            
#         self.num_sample = num_sample
#         self.data_samples = data_samples
#         self.fisher_matrix = self.compute_fisher()
        
#     def compute_fisher(self):
#         if isinstance(self.prior_model,  tf.keras.Sequential):
#             model = self.prior_model
#         else:
#             model = self.prior_model.sequential
               
#         for data in self.data_samples.shuffle(self.num_sample):
#             # Unpack the data.
#             x, y = data
            
#             with tf.GradientTape() as tape:
#                 y_pred = model(x, training=True)  # Forward pass
                
#                 # Compute our own loss
#                 loss = tf.keras.losses.mean_squared_error(y, y_pred)
                
#             # Compute gradients
#             trainable_vars = model.trainable_variables
#             gradients  = tape.gradient(loss, trainable_vars)
            
#             # Update weights
#             model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
#             # Update metrics (includes the metric that tracks the loss)
#             model.compiled_metrics.update_state(y, y_pred)
        
#         return model.get_weights()
        
        
#     def get_fisher(self):
#         return self.fisher_matrix




class EWC(object):
    
    def __init__(self, prior_model, data_samples, num_sample=30):
        self.num_sample = num_sample
        self.data_samples = data_samples        
        self.prior_model = prior_model
        if not isinstance(self.prior_model,  tf.keras.Sequential):
            self.prior_model = self.prior_model.sequential
                
        self.prior_weights = self.prior_model.weights        
        self.fisher_matrix = self.compute_fisher()
                
    def compute_fisher(self):
        model = self.prior_model
        weights = self.prior_weights
        fisher_accum = np.array([np.zeros(layer.numpy().shape) for layer in weights], 
                           dtype=object
                          )
        for data in self.data_samples.shuffle(self.num_sample):
            # Unpack the data.
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)  # Forward pass                
            grads = tape.gradient(y_pred, weights)
            for m in range(len(weights)):
                fisher_accum[m] += np.square(grads[m])
        fisher_accum /= self.num_sample
        return fisher_accum
    
    def get_fisher(self):
        return self.fisher_matrix
    