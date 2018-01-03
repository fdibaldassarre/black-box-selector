#!/usr/bin/env python3

import sys
import os
import argparse

path = "./image.png"
output_path = "./image_new.png"

parser = argparse.ArgumentParser(description="BlackBox predict")
parser.add_argument('input_image',default=None,
                    help='Input image')
parser.add_argument('output_image', default=None,
                    help='Output image')

args = parser.parse_args()

input_path = args.input_image
output_path = args.output_image

if input_path is None or output_path is None:
    print("Missing arguments")
    sys.exit(1)

if not os.path.exists(input_path):
    print("Input image does not exist")
    sys.exit(2)

print("Input: " + input_path)

from src import Model
from src.Generator import FiniteGenerator
from src.Places import MODELS_FOLDER
from src.Settings import IMAGE_WIDTH
from src.Settings import IMAGE_HEIGHT

print("Load model")
model_path = os.path.join(MODELS_FOLDER, "model.h5")
model = Model.initialize()
model.load(model_path)
# Load image data
generator = FiniteGenerator(input_path)
batch = generator.getBatch()
# Predict
result = model.predict_on_batch(batch)
# Reshape
result = result.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH))
# Convert batch to image
image = generator.recomposeImageFromSplits(result)
# Save
image.save(output_path)
