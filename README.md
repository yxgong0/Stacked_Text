# Stacked Text

### Dataset Download:
Baidu Netdisk: https://pan.baidu.com/s/1g_tZ6GZe2GoCl0TM5eYhMA?pwd=1234 Code: 1234

### Requirements

Python3
PyTorch
TorchVision
Pillow
imageio
numpy

### Usage
If GPU is available, the code wil automatically use the GPU.
To train a baseline model, use

```shell
python train.py --method none
```
To train a model with our method, use
```shell
python train.py --method test
```

To generate samples and test the metrics with the trained model, run
```shell
python generate_and_eval.py --method none
python generate_and_eval.py --method test
