## 3.1.0 (2020-09-29)

### Feat

- **train**: discord notification on train finished

### Fix

- added test loop

## 3.0.0 (2020-09-28)

### Feat

- **unet_model**: added ReduceLROnPlateau
- **unet_model**: added lr-scheduler
- metric update and bug fixes
- **train.py**: added accumulative grad and grad clipping
- **train.py**: re-wrote entire training loop
- **pre_processing**: added rescaling switch
- **dataset**: added data augmentation switch

### Fix

- fixed env variable for directories

## 2.0.0 (2020-09-25)

### Refactor

- removed .env file
- **unet**: moved arguments
- **lightning**: refractored pytorch code to pytorch-lightning
- black refractor
- black refractor
- **pre_processing**: duplication fix and type checking

### BREAKING CHANGE

- Complete refractor.
- Complete re-write of main training-loop.
- Random- Brightness, Saturation and Contrast replaced by RandomColorJitter.

### Perf

- **train**: non blocking on images

## 1.0.0 (2020-09-23)

### Perf

- **train.py**: enabled cudnn algorithm optimization

### Feat

- **tensorboard**: added validation loss
- **train.py**: added graph logging to tensorboard
- initial commit
