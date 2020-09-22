# Semantic Image Segmentation
**Members** : <a href="https://github.com/vegovs">Vegard Bergsvik Øvstegård</a>

**Supervisors** : <a href="https://www.mn.uio.no/ifi/personer/vit/jimtoer/">Jim Tørresen</a>

## Description

This repository aims to implement semantic image segmentation for [orthopohots](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/orthophoto).

## Installation(Linux)

```bash
git clone https://github.com/gil-uav/semantic-image-segmentation.git
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

```bash
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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT Licence](https://github.com/gil-uav/semantic-image-segmentation/blob/master/LICENSE)
