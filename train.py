#!/usr/bin/env python3

import os

from src import Model
from src.Places import MODELS_FOLDER
from src.Places import DATA_FOLDER
from src.Generator import TrainGenerator

print("Initialize model")
model = Model.initialize()
model_path = os.path.join(MODELS_FOLDER, "model.h5")
# Check if old models exist
if os.path.exists(model_path):
    print("Load model")
    model.load(model_path)
# Train
print("Train model")
train_data = os.path.join(DATA_FOLDER, "train/")
validation_data = os.path.join(DATA_FOLDER, "validation/")
generator_train = TrainGenerator(train_data)
generator_validation = TrainGenerator(validation_data)
model.fit_generator(generator_train, generator_validation)
print("Save")
model.save(model_path)
