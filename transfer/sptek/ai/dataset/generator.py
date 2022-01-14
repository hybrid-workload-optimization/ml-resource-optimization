import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf


def ylim(value, offset=1):
    y = []
    for x in value:
        y.extend(tf.reshape(x, [-1]).numpy())
    return (min(y) - offset, max(y) + offset)

def split_data(df, t=0.7, v=0.2):
    n = len(df)
    train = df[0:int(n*t)]
    val = df[int(n*t):int(n*(t+v))]
    test = df[int(n*(t+v)):]
    return train, val, test

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None, batch_size=32):
        
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        
        # get label of columns
        dataset = None
        if train_df is not None:
            dataset = train_df
        if val_df is not None:
            dataset = val_df
        if test_df is not None:
            dataset = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(dataset.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]    
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col='cpu_avg.', max_subplots=3, subplots=None, samples=None, figsize=(12, 8)):

        if samples is not None:
            inputs, labels = samples
        else:
            inputs, labels = self.example

        input_size = len(inputs)
        plot_col_index = self.column_indices[plot_col]

        max_n = 0
        if subplots is not None:
            max_n = len(subplots)
        max_n = min(max(max_n, max_subplots), input_size)

        if subplots is None:
            slots = random.sample(range(input_size), max_n)
        else:
            slots = subplots

        a = tf.reshape(inputs, [-1]).numpy()
        b = tf.reshape(labels, [-1]).numpy()
        #print(a, b)
        bottom, top = ylim([a, b])

        plt.figure(figsize=figsize)
        for index in range(max_n):
            n = slots[index]
            plt.subplot(max_n, 1, index+1)
            plt.ylim(bottom, top)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(self.label_indices, labels[n, :, label_col_index],
                     label='Labels', marker='.', zorder=-10, c='#2ca02c')

            if model is not None:
                predictions = model(inputs)
                plt.plot(self.label_indices, predictions[n, :, label_col_index],
                            label='Predictions', marker='.', zorder=-10, c='#ff7f0e')

            if n == 0:
                plt.legend()

            IN = mpatches.Patch(color='blue', label='Inputs')
            LB = mpatches.Patch(color='#2ca02c', label='labels')
            PR = mpatches.Patch(color='#ff7f0e', label='predictions')
            plt.legend(handles=[IN,LB,PR], loc=2)

        plt.xlabel('Time [h]')
        return slots
    
    def make_dataset(self, data):
        if len(data) < 1:
            #return data
            return None

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        return next(iter(self.test))
    
    def get_input(self, dataset):
        # return iter(dataset).get_next()
        try:
            return next(iter(dataset))
        
        except StopIteration:
            return dataset
    
    def get_input_val(self, dataset):
        try:
            _input_, _ = self.get_input(dataset)
        except:
            _input_, _ = dataset
        return _input_.numpy()

    def copy(self):
        return WindowGenerator(
            self.input_width,
            self.label_width,
            self.shift,
            self.train_df.copy(),
            self.val_df.copy(),
            self.test_df.copy(),
            self.label_columns,
            self.batch_size )
