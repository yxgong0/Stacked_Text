# Stacked Text

### Dataset Download:

Stacked_Text:

Baidu Netdisk: https://pan.baidu.com/s/1g_tZ6GZe2GoCl0TM5eYhMA?pwd=1234 Code: 1234

ICDAR13_chars:

Baidu Netdisk: https://pan.baidu.com/s/1zuv09vkjd27_hsCLZF5AIg?pwd=1234 Code: 1234

### Requirements

Python3
PyTorch
TorchVision
Pillow
imageio
numpy

### Usage
Create ./data folder, and unzip the dataset file to this folder. The default code will utilize the static dataset Stacked_Text; if you want to generate the stacked images dynamically in training, download and unzip the ICDAR13_chars file, and change the dataset class in train.py from StackedText_static to StackedText.

If GPU is available, the code will automatically use the GPU.
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
