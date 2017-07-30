# Black box selector

Find transparent black boxes in an image.

## Installation requirements

- Python 3
- numpy
- PIL
- scikit-image
- Keras

## Usage

```sh
./predict.py image.png output.png
```

## Examples

Input image (64x64):

![input](https://raw.githubusercontent.com/fdibaldassarre/black-box-selector/master/examples/input64.png)

Result:

![input](https://raw.githubusercontent.com/fdibaldassarre/black-box-selector/master/examples/output64.png)

The program works with images of any size by averaging the result on multiple overlapping 64x64 areas.

Input image:

![input](https://raw.githubusercontent.com/fdibaldassarre/black-box-selector/master/examples/input.png)

Result:

![input](https://raw.githubusercontent.com/fdibaldassarre/black-box-selector/master/examples/output.png)

## Training

### Training data

Put some clean images in data/train and data/validation.

### NN setup

The parameters of the neural network are in src/Settings.py

### Training

Run
```sh
./train.py
```

The training took a couple of hours on my laptop using a 920MX nVidia card (via optimus) with Tensorflow backend.
