import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sagemaker
from sagemaker import RandomCutForest

if __name__ == "__main__":
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