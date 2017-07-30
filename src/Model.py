#!/usr/bin/env python3

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Dropout

from keras.models import Model

from .Settings import IMAGE_WIDTH
from .Settings import IMAGE_HEIGHT
from .Settings import CATEGORIES
from .Settings import CHANNELS
from .Settings import PREPROCESSED_LAYERS
from .Settings import BATCH_SIZE
from .Settings import BATCH_SIZE_VALIDATION
from .Settings import EPOCHS
from .Settings import PRETRAIN_EPOCHS
from .Settings import STEPS_PER_EPOCH


class BlackBoxR():

    def __init__(self):
        inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS + PREPROCESSED_LAYERS, ))
        # 3-Conv + MaxPooling
        conv_0 = inputs
        conv_1 = Conv2D(16, kernel_size=(3,3), activation='relu')(conv_0)
        conv_2 = self._add3ConvAndMaxPool(32)(conv_1)
        conv_3 = self._add3ConvAndMaxPool(64)(conv_2)
        conv_4 = self._add3ConvAndMaxPool(128)(conv_3)
        conv_5 = self._add3ConvAndMaxPool(256)(conv_4)
        dense_0 = Flatten()(conv_5)
        dense_1 = Dense(2048, activation='relu')(dense_0)
        dense_1b = Dropout(0.5)(dense_1)
        dense_2 = Dense(1024, activation='relu')(dense_1b)
        deconv_0 = Reshape(target_shape=(32, 32, 1))(dense_2)
        deconv_1 = Conv2DTranspose(1, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(deconv_0)
        prediction = Flatten()(deconv_1)
        '''
        conv_2 = Conv2D(128, kernel_size=(3,3), activation='relu')(conv_1)
        mp_1 = MaxPool2D(pool_size=(2,2))(conv_2)
        conv_3 = Conv2D(256, kernel_size=(3,3), activation='relu')(mp_1)
        mp_2 = MaxPool2D(pool_size=(2,2))(conv_3)
        conv_4 = Conv2D(512, kernel_size=(3,3), activation='relu')(mp_2)
        mp_3 = MaxPool2D(pool_size=(2,2))(conv_4)
        # Dense
        dense_0 = Flatten()(mp_3)
        dense_0b = Dropout(0.5)(dense_0)
        dense_1 = Dense(4096, activation='relu')(dense_0b)
        prediction = Dense(IMAGE_WIDTH*IMAGE_HEIGHT, activation='linear')(dense_1)
        '''
        #dense_2 = Dense(IMAGE_WIDTH*IMAGE_HEIGHT, activation='relu')(dense_1b)
        # prediction = Reshape(target_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))(dense_2)
        # Deconvolution
        '''
        dense_2 = Dense(64, activation='relu')(dense_1b)
        deconv_0 = Reshape(target_shape=(2,2,16))(dense_2)
        deconv_1 = Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(deconv_0)
        deconv_2 = Conv2DTranspose(4, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(deconv_1)
        deconv_3 = Conv2DTranspose(2, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(deconv_2)
        deconv_4 = Conv2DTranspose(1, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(deconv_3)
        prediction = deconv_4
        '''
        self.model = Model(inputs, prediction)
        self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    def _add3ConvAndMaxPool(self, size):
        def add(layer):
            conv = Conv2D(size, kernel_size=(3,3), activation='relu')(layer)
            max_pool = MaxPool2D(pool_size=(2,2))(conv)
            return max_pool
        return add

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)

    def fit_generator(self, generator_train, generator_validation):
        self.model.fit_generator(
                        generator_train.flow(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generator_validation.flow(batch_size=BATCH_SIZE_VALIDATION),
                        validation_steps=1,
                        epochs=EPOCHS,
                        verbose=1)

    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch)

def initialize(*args, **kwargs):
    return BlackBoxR(*args, **kwargs)
