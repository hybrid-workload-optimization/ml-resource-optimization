import os
import numpy as np
import pandas as pd
import tensorflow as tf

class mnist(object):

    def __init__(self):
        self.name = 'mnist'
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

    def __repr__(self):
        return "\n".join([
            f"MNIST DATABASE",
            f"train set images: {self.train_images.shape[0]}",
            f"test set images:{self.test_images.shape[0]}",
            f"total set images: {self.train_images.shape[0]+self.test_images.shape[0]}"            
            f"\n"
        ])


    def preprocess(self, sample_size=None):
        if sample_size is None:
            train_size = self.train_images.shape[0]
            test_size = self.test_images.shape[0]


        self.train_labels = self.train_labels[:train_size]        
        self.test_labels = self.test_labels[:test_size]
        
        self.train_images = self.train_images[:train_size].reshape(-1, 28 * 28) / 255.0
        print(f"preprocess train images ({train_size})")

        self.test_images = self.test_images[:test_size].reshape(-1, 28 * 28) / 255.0
        print(f"preprocess test images ({test_size})\n")


class climate(object):

    def __init__(self):
        self.name = 'climate'
        self.origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
        zip_path = tf.keras.utils.get_file(
            origin=self.origin,
            fname='jena_climate_2009_2016.csv.zip',
            extract=True)
        self.csv_path, _ = os.path.splitext(zip_path)
        self.data = pd.read_csv(self.csv_path)
        print(self.data.info())
        
    def __repr__(self):
        return "\n".join([
            f"CLIMATE DATABASE",
            f"origin: {self.origin}",
            f"file: {self.csv_path}",
            f"\n"
        ])


    def preprocess(self, sample_size=None):
        df = self.data[5::6]
        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

        print(f"Remove outliers of Wind velocity.")
        wv = df['wv (m/s)']
        bad_wv = wv == -9999.0
        wv[bad_wv] = 0.0
        
        max_wv = df['max. wv (m/s)']
        bad_max_wv = max_wv == -9999.0
        max_wv[bad_max_wv] = 0.0


        print(f"Feature engineering to model wind data")
        wv = df.pop('wv (m/s)')
        max_wv = df.pop('max. wv (m/s)')

        wd_rad = df.pop('wd (deg)')*np.pi / 180

        df['Wx'] = wv*np.cos(wd_rad)
        df['Wy'] = wv*np.sin(wd_rad)

        df['max Wx'] = max_wv*np.cos(wd_rad)
        df['max Wy'] = max_wv*np.sin(wd_rad)

        print(f"Feature engineering to model time data")
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        day = 24*60*60
        year = (365.2425)*day

        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        self.data = df

        print(f"Split the data.")
        df = self.data
        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]
        
        print(f"Normalize the data.")
        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train = (train_df - train_mean) / train_std
        self.val = (val_df - train_mean) / train_std
        self.test = (test_df - train_mean) / train_std
    
        IN_STEPS = 24
        OUT_STEPS = 1
        self.window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=self.train,
                               val_df=self.val,
                               test_df=self.test,
                               label_columns=['T (degC)'])

        self.units = 32
        self.in_steps = IN_STEPS
        self.out_steps = OUT_STEPS
        self.in_features = df.shape[1]
        self.out_features = 1

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

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
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            # batch_size=32,)
            batch_size = int(self.input_width / 2),)

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
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result