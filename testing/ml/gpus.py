import tensorflow as tf

_gpu_size_ = 1024

def set_device_configuration(device=0, size=None):
    global _gpu_size_
    if size:        
        _gpu_size_ = size
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[device],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= _gpu_size_ )])
        except RuntimeError as e:
            print(e)

