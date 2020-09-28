# Semantic Image Segmentation
**Members** : <a href="https://github.com/vegovs">Vegard Bergsvik Øvstegård</a>

**Supervisors** : <a href="https://www.mn.uio.no/ifi/personer/vit/jimtoer/">Jim Tørresen</a>

## Status
![Python package](https://github.com/gil-uav/semantic-image-segmentation/workflows/Python%20package/badge.svg?branch=master) ![Python Package using Conda](https://github.com/gil-uav/semantic-image-segmentation/workflows/Python%20Package%20using%20Conda/badge.svg) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/41c8bd8d0049413a9432bdd78e4e3869)](https://www.codacy.com/gh/gil-uav/semantic-image-segmentation/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gil-uav/semantic-image-segmentation&amp;utm_campaign=Badge_Grade) [![DeepSource](https://deepsource.io/gh/gil-uav/semantic-image-segmentation.svg/?label=active+issues&show_trend=true)](https://deepsource.io/gh/gil-uav/semantic-image-segmentation/?ref=repository-badge) [![DeepSource](https://deepsource.io/gh/gil-uav/semantic-image-segmentation.svg/?label=resolved+issues&show_trend=true)](https://deepsource.io/gh/gil-uav/semantic-image-segmentation/?ref=repository-badge)

## Description

This repository aims to implement and produce trained networks in semantic image segmentation for
[orthopohots](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/orthophoto).
Current network structure is [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Dependencies
* [Python](https://www.python.org/) (version 3.6, 3.7 or 3.8)
* [Pip](https://virtualenv.pypa.io/en/latest/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/) or:
* [conda](https://docs.conda.io/en/latest/)
* [Cuda](https://developer.nvidia.com/cuda-10.2-download-archive) version 10.2

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
The application fetches some configurations and parameters from a .env file if it exists.
Run `python train.py --help` to see all other arguments. The package is using [pytorch-lighting](https://github.com/PyTorchLightning/pytorch-lightning) and inherits all its arguments.

The data is expected to be structured like this:
```
data/
    images/
    masks/
```
The path do data us set using --dp argument.

#### Console example
This example stores the checkpoints and logs under the default_root_dir, uses all available GPUs and
fetches training data from --dp.

```console
python train.py --default_root_dir=/shared/use/this/ --gpus=-1 --dp=/data/is/here/
```

#### .env example
Only these arguments are fetched from .env, the rest must be passed through the CLI.
```
# Model config
N_CHANNELS=3
N_CLASSES=1
BILINEAR=True

# Hyperparameters
EPOCHS=300 # Epochs
BATCH_SIZE=4 # Batch size
LRN_RATE=0.001 # Learning rate
VAL_PERC=15 # Validation percent
TEST_PERC=15 # Testing percent
IMG_SIZE=512  # Image size
VAL_INT=1 # Validation interval
ACC_GRAD=4 # Accumulated gradients, number = K.
GRAD_CLIP=1.0 # Clip gradients with norm above give value
EARLY_STOP=10 # Early stopping patience(Epochs)

# Other
PROD=False # Turn on or off debugging APIs
DIR_DATA="data/" # Where dataset is stored
DIR_ROOT_DIR="/shared/use/this/" # Where logs and checkpoint will be stored
WORKERS=4 # Number of workers for data- and validation loading
```

### Performance tips:
* Try with different number of workers, but more than 0. A good starting point
is `workers = cores * (threads per core)`.
* Install Pillow-SIMD as described in [Installation](#installation).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT Licence](https://github.com/gil-uav/semantic-image-segmentation/blob/master/LICENSE)
