import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sagemaker
from sagemaker import RandomCutForest
from ai.utils import path as pathutil
from ai.master import Master
from pathlib import Path

def test_sagemaker():
    data_filename = '/root/dev/cmpai-v3-anomaly/data/sample/nyc_taxi.csv'
    taxi_data = pd.read_csv(data_filename, delimiter=",")
    print(taxi_data.head())


    matplotlib.rcParams["figure.dpi"] = 100
    # taxi_data.plot()
    taxi_data[5500:6500].plot()
    plt.show()
    plt.savefig('./temp/boston.png')

    execution_role = sagemaker.get_execution_role()

    session = sagemaker.Session()

    # specify general training job information
    rcf = RandomCutForest(
        role=execution_role,
        instance_count=1,
        train_instance_count=1,
        train_instance_type='local_gpu',
        num_samples_per_tree=512,
        num_trees=50,
    )

    # automatically upload the training data to S3 and run the training job
    rcf.fit(rcf.record_set(taxi_data.value.to_numpy().reshape(-1, 1)))
    
def test_uri():
    path = '/root/dev/cmpai-v3-anomaly/mastor-anomaly-ver1.yml'
    master = Master(path)
    master.load()
    model = master.import_model(dtype='sequential')
    uri = model.uri
    normpath = pathutil.uri_to_normpath(uri)
    absolutized = pathutil.uri_to_absolutized(uri)
    print(f"(debug) normpath -> {normpath}")
    print(f"(debug) absolutized -> {absolutized}")    
    print(f"(debug) normpath root -> {Path(normpath).root}")
    print(f"(debug) absolutized root -> {Path(absolutized).root}")    
    print(f"(debug) './test' root -> {Path('./test').root}")
    print(f"(debug) '../reat' root -> {Path('../test').root}")
    
    
    
if __name__ == "__main__":
    # test_sagemaker()
    test_uri()