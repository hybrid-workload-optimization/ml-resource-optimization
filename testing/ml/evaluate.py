import sys
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import gpus
import callback
import datasets
import model
import recovery


def tf_print(msg):
    print(f"tensorflow {msg}")

# check tf log level
def check_tf_log_disable():
    return 'disable' if os.environ['TF_CPP_MIN_LOG_LEVEL'] == '3' else 'enable'

# load dataset
def load_dataset(dataset):
    if dataset == 'mnist':
        return datasets.mnist()

    if dataset == 'climate':
        return datasets.climate()
        

# create tf model
def create_tf_model(dataset):
    if dataset.name == 'mnist':
        return model.mnist(dataset)

    if dataset.name == 'climate':
        return model.climate(dataset)

#  create checkpoint
def create_checkpoint(path):
    checkpoint_path = Path(path)
    if not checkpoint_path.parent.exists():
        checkpoint_path.parent.mkdir()
    return callback.Checkpoint(checkpoint_path)

    


# main function
def main():

    # check flag
    if len(sys.argv) > 1:
        flag = sys.argv[1]
        
        # check interrupt flag
        if flag == '-interrupt':
            # create checkpoint path
            checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
            checkpoint = create_checkpoint(path=checkpoint_path)
            recovery_point = recovery.Recovery(checkpoint, None)
            print("")
            tf_print(f"-- interrupt description.\n{recovery_point.interrupt_history()}")
            return

        # check interrupt flag
        if flag == '-recovery':
            # create checkpoint path
            checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
            checkpoint = create_checkpoint(path=checkpoint_path)
            recovery_point = recovery.Recovery(checkpoint, None)
            print("")
            tf_print(f"-- recovery description.\n{recovery_point.recovery_history()}\n")
            print(f"{recovery_point}")
            return

    print("")
    tf_print(f"version: {tf.version.VERSION}")
    tf_print(f"device : {tf.test.gpu_device_name()}")
    tf_print(f"logger : {check_tf_log_disable()}\n")
    
    # initailize gpu
    gpus.set_device_configuration(size=256)
    
    # load dataset
    dataset_name = 'mnist'
    epochs = 100
    # dataset_name = 'climate'
    dataset = load_dataset(dataset_name)
    tf_print(f"-- load datasets '{dataset_name}'.\n{dataset}")

    # preprocess dataset
    tf_print(f"-- preprocess dataset '{dataset_name}'.")
    dataset.preprocess()

    # # create checkpoint
    checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint = create_checkpoint(path=checkpoint_path)
    tf_print(f"-- create checkpoint.\n{checkpoint}\n")

    # create tensorflow model
    tf_print(f"-- create machine learning model {dataset.name}.\n")
    model = create_tf_model(dataset)
    model.compile()

    # create recoevery
    tf_print(f"-- create recoevery machine learning model {dataset.name}.")
    recovery_point = recovery.Recovery(checkpoint, model.metrics)
    if recovery_point.is_recovery():
        recovery_point.restore(model)
        print(f"{recovery_point}\n")

    print(recovery_point.history())
    print("")

    # build machine learning mocdel
    tf_print(f"-- build machine learning model {dataset.name}.")
    model.fit(checkpoint, recovery_point, epochs)

    # report machine learning mocdel
    _, model_acc = model.evaluate()
    print("\nAccuracy of the model: {:5.2f}%".format(100*model_acc))

    recovery_point.flush()

    # save plot of ml model
    save_ml_plot(model)


def save_ml_plot(model):
    history = model.history()
    axis_loss = np.array(history['loss'])
    axis_loss = np.append(axis_loss, history['val_loss'], axis=0)
    axis_acc = np.array(history[model.metrics[0]])
    axis_acc = np.append(axis_acc, history[model.metrics[1]], axis=0)

    fig, ax = plt.subplots(2, figsize=(10, 6))

    ax[0].set_xlabel('')
    ax[1].set_xlabel('epochs')

    ax[0].set_ylabel('loss ', size=10)
    ax[1].set_ylabel(f"{model.metrics[0]}", size=10)
    ax[0].set_title('history of train machine learing', size=14)
    ax[0].set_xticks(np.arange(0, axis_loss.size, 1))
    ax[1].set_xticks(np.arange(0, axis_acc.size, 1))
    ax[0].xaxis.set_ticklabels([str(i+1) for i in range(axis_acc.size)])
    ax[1].xaxis.set_ticklabels([str(i+1) for i in range(axis_acc.size)])

    shift = 0.05
    ax[0].set_ylim(axis_loss.min() - shift, axis_loss.max() + shift)
    ax[1].set_ylim(axis_acc.min() - shift, axis_acc.max() + shift)

    ax[0].plot(history['loss'], color='0.5', alpha=0.8, label='train')
    ax[0].plot(history['val_loss'], color='#E8685D', alpha=0.8, label='test')
    # ax[0].legend(['train', 'test'], loc='upper right');
    ax[0].legend(['train', 'test'], loc='upper right', frameon=True, fontsize=10)

    ax[1].plot(history[model.metrics[0]], color='0.5', alpha=0.8, label='train')
    ax[1].plot(history[model.metrics[1]], color='#E8685D', alpha=0.8, label='test')
    # ax[1].legend(['train', 'test'], loc='upper right');
    ax[1].legend(['train', 'test'], loc='upper right', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig('./fig/evaluate.png', dpi=1200)



def save_ml_plot_2(model):
    history = model.history()
    axis_loss = np.array(history['loss'])
    axis_loss = np.append(axis_loss, history['val_loss'], axis=0)
    axis_loss_min = axis_loss.min()
    axis_loss_max = axis_loss.max()

    axis_acc = np.array(history['accuracy'])
    axis_acc = np.append(axis_acc, history['val_accuracy'], axis=0)
    axis_acc_min = axis_acc.min()
    axis_acc_max = axis_acc.max()


    plt.subplot(2,1,1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.grid(True)  
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim([axis_loss_min - 0.05, axis_loss_max + 0.05])
    plt.legend(['train', 'test'], loc='upper right');

    plt.subplot(2,1,2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.grid(True)  
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim([0, axis_acc_max + 0.05])
    plt.legend(['train', 'test'], loc='upper right');

    plt.tight_layout()
    plt.show()
    plt.savefig('./fig/evaluate.png', dpi=1200)





def test(dataset, checkpoint):
    # check recovery flag
    model = create_tf_model(dataset)
    model.compile()    
    recovery_point = recovery.Recovery(checkpoint)
    if recovery_point.is_recovery():
        recovery_point.restore(model)
        tf_print(f"-- create recovery point.\n{recovery_point}\n")
        tf_print(f"-- Describe of the restored model.\n{model}\n")
        print("\nrestored ml model completed...")
        recovery_point.flush()

    else:
        # check recovery flag
        tf_print(f"-- create tensorflow model {dataset.name}.")
        train_model = create_tf_model(dataset)
        train_model.compile_fit(checkpoint, recovery_point)
        _, train_acc = train_model.evaluate()
        print("Accuracy of the model: {:5.2f}%".format(100*train_acc))
        print("\ncreate ml model completed...")

if __name__ == "__main__":
    main()
    print("")