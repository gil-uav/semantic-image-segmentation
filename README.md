# Semantic Image Segmentation
**Members** : <a href="https://github.com/vegovs">Vegard Bergsvik Øvstegård</a>

**Supervisors** : <a href="https://www.mn.uio.no/ifi/personer/vit/jimtoer/">Jim Tørresen</a>

## Status
![Python package](https://github.com/gil-uav/semantic-image-segmentation/workflows/Python%20package/badge.svg?branch=master) ![Python Package using Conda](https://github.com/gil-uav/semantic-image-segmentation/workflows/Python%20Package%20using%20Conda/badge.svg) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/41c8bd8d0049413a9432bdd78e4e3869)](https://www.codacy.com/gh/gil-uav/semantic-image-segmentation/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gil-uav/semantic-image-segmentation&amp;utm_campaign=Badge_Grade)

## Description

This repository aims to implement and produce trained networks in semantic image segmentation for
[orthopohots](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/orthophoto).
Current network structure is [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Dependencies
* [Python](https://www.python.org/) (version 3.6 or 3.7)
* [Pip](https://virtualenv.pypa.io/en/latest/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/) (Recommended build-method)

## Installation

```console
git clone https://github.com/gil-uav/semantic-image-segmentation.git
```

#### virtualenv

```console
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### Conda
```console
conda env create --file environment.yml
conda activate seg
```

Uninstall Pillow and install Pillow-SIMD:
```console
pip uninstall pillow
pip install pillow-simd
```
If you have a AVX2-enabled CPU, check with `grep avx2 /proc/cpuinfo`, you can install Pillow-SIMD with:
```console
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```
This should have slightly better performance than the SSE4(default) version and much better than the standard Pillow
 package.

NB! Remember to uninstall and reinstall Pillow-SIMD. In some cases, python might not find the PIL
package, however a reinstall fixes this 99% of the time.

## Usage

### Training
The application fetches configurations and parameters from a .env file if it exists.
This can be overridden and changed by passing the following arguments:

```console
usage: train.py [-h] [-e E] [-b [B]] [-lr [LR]] [-f LOAD] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs
  -b [B], --batch-size [B]
                        Batch size
  -lr [LR], --learning-rate [LR]
                        Learning rate
  -f LOAD, --load LOAD  Load model from a .pth file
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
```

#### .env example
```
# U-net model config
N_CHANNELS=3 # E.g RGB
N_CLASSES=1
BILINEAR=True # Upsampling

# Hyperparameters
E=10 # Epochs
BS=4 # Batch size
LR=0.0001 # Learning rate
VAL=10.0 # Validation percent
IS=700  # Image size

# Other
PROD=True # Turn on or off debugging APIs
DIR_IMG="data/imgs/"
DIR_MASK="data/masks/"
DIR_CHECKPOINTS="checkpoints/"
WORKERS=8 # Number of workers for data- and validation loading
```

## Optimizations
- [x] Asynchronous data loading and augmentation enabled.
- [x] cuDNN autotuner enabled.
- [x] Automatic mixed precision, i.e use of float16 in tensorcores.
- [ ] Increase batch-size due to reduced memory usage.
    - [ ] Tune learning rate.
    - [ ] Add learning rate warmup.
    - [x] Add learning decay.
    - [ ] Add weight decay.
    - [ ] Use optimizer for large-barch training(LARS, LAMB, NVLAMB, NovoGrad).
- [x] Disable bias for convolutions directly followed by a batch norm.
- [x] Use parameter.grad = None.
- [x] Disbale debut APIs in production-mode.
- [ ] Opt for [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
instead of [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) to improve Multi-GPU efficiency.
    - [ ] Load-balance workload on multiple GPUs.
- [ ] Use fused building blocs from [APEX](https://github.com/NVIDIA/apex).
- [ ] Checkpoint to recompute intermediates.
- [ ] Fuse pointwise operations.

### Performance tips:
* Try with different number of workers, but more than 0. A good starting point
is `workers = cores * (threads per core)`.
* Install Pillow-SIMD as described in [Installation](#installation).
* Make sure you have a GPU with Cuda version 10.2.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT Licence](https://github.com/gil-uav/semantic-image-segmentation/blob/master/LICENSE)
