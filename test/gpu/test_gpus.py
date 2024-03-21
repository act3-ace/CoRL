def test_pytorch_setup():
    import torch

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
