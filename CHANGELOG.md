## 2.0.0 (2020-09-25)

### Perf

- **train**: non blocking on images

### Feat

- **train.py**: re-wrote entire training loop
- **pre_processing**: added rescaling switch
- **dataset**: added data augmentation switch

### BREAKING CHANGE

- Complete re-write of main training-loop.
- Random- Brightness, Saturation and Contrast replaced by RandomColorJitter.

### Refactor

- black refractor
- **pre_processing**: duplication fix and type checking

## 1.0.0 (2020-09-23)

### Perf

- **train.py**: enabled cudnn algorithm optimization

### Feat

- **tensorboard**: added validation loss
- **train.py**: added graph logging to tensorboard

## 0.1.0 (2020-09-22)

### Feat

- initial commit
