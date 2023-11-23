import os
import tensorflow as tf
from tensorflow import keras

class mnist(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10)
        ])
        self._history_ = None
        self.metrics = ['accuracy','val_accuracy']

    def load_weights(self, path):
        self.model.load_weights(path).expect_partial()

    def compile(self):
        self.model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    def fit(self, checkpoint, recovery_point, epochs=10):
        # self.model.save_weights(os.fspath(checkpoint.checkpoint_path).format(epoch=0))
        recovery_point.max_epochs(epochs)

        # 새로운 콜백으로 모델 훈련하기
        self._history_ = self.model.fit(self.dataset.train_images,
                                    self.dataset.train_labels,
                                    initial_epoch=recovery_point.epochs(),
                                    epochs=epochs,
                                    validation_data=(self.dataset.test_images, self.dataset.test_labels),
                                    callbacks=[checkpoint.callback, recovery_point.callback])  # 콜백을 훈련에 전달합니다

    def compile_fit(self, checkpoint, recovery_point, epochs=10):
        self.compile()
        self.fit(checkpoint, recovery_point, epochs)

    def evaluate(self, verbose=2):
        return self.model.evaluate(self.dataset.test_images, self.dataset.test_labels, verbose=verbose)

    def history(self):
        if self._history_:
            return self._history_.history

    def __call__(self, data):
        pass
    
    def __repr__(self):
        _, train_acc = self.evaluate()
        return '\n'.join([
            "Accuracy of the model: {:5.2f}%".format(100*train_acc)
        ])


class climate(object):
    def __init__(self, dataset):        
        self.dataset = dataset
        self.model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(dataset.units, return_sequences=False, activation="tanh"), input_shape=(dataset.in_steps, dataset.in_features)),
            tf.keras.layers.Dense(dataset.out_steps * dataset.out_features,
                                kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([dataset.out_steps, dataset.out_features])
        ])

        self._history_ = None
        self.metrics = ['mean_absolute_error','val_mean_absolute_error']

    def load_weights(self, path):
        self.model.load_weights(path).expect_partial()

    def compile(self):        
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
    
    def fit(self, checkpoint, recovery_point, epochs=10):
        print(f"train element spec: {self.dataset.window.train.element_spec}\n")
        recovery_point.max_epochs(epochs)
        self._history_ = self.model.fit(self.dataset.window.train,
                            epochs=epochs,
                            initial_epoch=recovery_point.epochs(),
                            validation_data=self.dataset.window.val,
                            callbacks=[checkpoint.callback, recovery_point.callback])

    def compile_fit(self, checkpoint, recovery_point, epochs=10):
        self.compile()
        self.fit(checkpoint, recovery_point, epochs)

    def evaluate(self, verbose=2):
        return self.model.evaluate(self.dataset.window.val)
        
    def history(self):
        if self._history_:
            return self._history_.history

    def __call__(self, data):
        pass
    
    def __repr__(self):
        _, train_acc = self.evaluate()
        return '\n'.join([
            "Accuracy of the model: {:5.2f}%".format(100*train_acc)
        ])