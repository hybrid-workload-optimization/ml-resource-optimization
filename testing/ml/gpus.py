import tensorflow as tf

# Golbal variable
_gpu_size_ = 1024

def set_device_configuration(device=0, size=None):
    global _gpu_size_
    if size:        
        _gpu_size_ = size
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[device],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= _gpu_size_ )])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)

