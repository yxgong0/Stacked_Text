## Implementation of Stacked Text experiments

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
```


It is worth noting that we use a static dataset with 10M images in the paper. However, we cannot provide the static dataset due to the file size limitation. Therefore, this code supports training with dynamic dataset (Read ICDAR 2013 character images and stack them while training).