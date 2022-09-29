import pytest

def test_tf_setup():
    import tensorflow as tf
    tf.config.list_physical_devices('GPU')
    sys_details = tf.sysconfig.get_build_info()
    cuda = sys_details["cuda_version"]
    cudnn = sys_details["cudnn_version"]
    print(f"{cuda=}, {cudnn=}")
    print(sys_details)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices())

def test_pytorch_setup():
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
