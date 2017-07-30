#!/usr/bin/env python3

import os
import time
import random
random.seed(time.time())

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
import numpy
from skimage.feature import canny

from src import ImageSplitter

from src.Settings import IMAGE_WIDTH
from src.Settings import IMAGE_HEIGHT
from src.Settings import CHANNELS
from src.Settings import PREPROCESSED_LAYERS
from src.Settings import OPACITY_MIN
from src.Settings import OPACITY_MAX
from src.Settings import OPACITY_PRETRAIN_MIN
from src.Settings import OPACITY_PRETRAIN_MAX
from src.Settings import BOX_SIZE_REL
from src.Settings import BATCH_SIZE
from src.Settings import STRIDE_WIDTH
from src.Settings import STRIDE_HEIGHT
from src.Settings import STRIDE_TRAIN_WIDTH
from src.Settings import STRIDE_TRAIN_HEIGHT

'''
    Generator
        Base generator
    Interface:
        - flow
        - getBatch
'''
class Generator():
    _source = None
    #_samples_for_epoch = 10
    _image_splitter = None

    def __init__(self, source):
        self._source = source
        self._image_splitter = ImageSplitter(CHANNELS)

    def getImageSplitter(self):
        return self._image_splitter

    '''
    def getSamplesPerEpoch(self):
        return self._samples_for_epoch

    def setSamplesPerEpoch(self, steps):
        self._samples_for_epoch = steps
    '''

    def flow(self):
        raise NotImplemented

    def getBatch(self):
        raise NotImplemented

'''
    ImageGenerator
        Image processing to feed a neural network.
    Implements:
        - getInputData
        - preprocessImage
'''
class ImageGenerator(Generator):

    '''
        preprocessImage
            Apply some kind of pre-processing to an image.
        Input:
            - img: a PIL.Image object
        Returns:
            - processed_image: numpy array
    '''
    def preprocessImage(self, img):
        processed_image = numpy.zeros(
                            (IMAGE_HEIGHT, IMAGE_WIDTH, PREPROCESSED_LAYERS),
                            dtype="float32")
        # Increase contrast
        contrast = ImageEnhance.Contrast(img)
        cimg = contrast.enhance(2)
        processed_image[:,:,0:CHANNELS] = numpy.asarray(cimg) / 255.
        # Compute edges
        img_g = cimg.convert('L')
        data = numpy.asarray(img_g, dtype="float32") / 255.
        edges = canny(data)
        processed_image[:,:,-1] = numpy.float32(edges)
        return processed_image

    '''
        modifyImage
            Modify an image.
        Input:
            - img: PIL.Image
        Returns:
            - result_image: modified PIL.Image
            - process: operation applied
    '''
    def modifyImage(self, img):
        return img, None


    '''
        getInputData
            Get the data to feed to the network.
        Input:
            - img: PIL.Image
        Returns:
            - data: numpy array
            - process: process applied to the image
    '''
    def getInputData(self, img):
        # Process the image
        base_img, process = self.modifyImage(img)
        base_data = self._image_splitter.getDataFromImage(img) / 255.
        # Pre-process the image
        processed_data = self.preprocessImage(img)
        # Put all together
        data = numpy.zeros(
                    (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS+PREPROCESSED_LAYERS),
                    dtype="float32")
        data[:, :, 0:CHANNELS] = base_data
        if PREPROCESSED_LAYERS > 0:
            data[:, :, CHANNELS:CHANNELS+PREPROCESSED_LAYERS] = processed_data
        return data, process

'''
    Wrong source type exception
'''
class WrongSourceType(BaseException):
    pass

'''
    FiniteGenerator:
        Generate all the splits for an image and then stop.
'''
class FiniteGenerator(ImageGenerator):

    def __init__(self, source):
        super().__init__(source)
        try:
            self._img = Image.open(source)
            self._split_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
            self._stride_size = (STRIDE_WIDTH, STRIDE_HEIGHT)
        except Exception:
            raise WrongSourceType

    def getBatch(self):
        batch = []
        for split_img in self._image_splitter.splitImage(
                                                self._img,
                                                self._split_size,
                                                padding=True,
                                                stride=self._stride_size):
            data, _ = self.getInputData(split_img)
            batch.append(data)
        return numpy.asarray(batch, dtype="float32")

    def recomposeImageFromSplits(self, splits, resolver=None):
        data = self._image_splitter.recomposeImage( splits,
                                                    self._img.size,
                                                    stride=self._stride_size,
                                                    resolver=resolver)
        img = self._image_splitter.getImageFromData(data)
        im_height, im_width = self._img.size
        return img.crop((0, 0, im_height, im_width))


'''
    Standardizer
        Replace low numbers with 0.0 and others with 1.0
'''

def standardizer(x):
    if x < 0.1:
        return 0
    else:
        return 1

standardizer_numpy = numpy.vectorize(standardizer)

'''
    TrainGenerator:
        Generate training examples
'''
class TrainGenerator(ImageGenerator):

    def __init__(self, source):
        super().__init__(source)
        if not os.path.exists(source) or not os.path.isdir(source):
            raise WrongSourceType
        self._folder = source
        self._files = os.listdir(self._folder)
        self._split_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        self._stride_size = (STRIDE_TRAIN_WIDTH, STRIDE_TRAIN_HEIGHT)

    '''
        getRectangleVertices
            Get 4 random vertices for a rectangle.
    '''
    def getRectangleVertices(self, image_size):
        width, height = image_size
        # Set box size
        # TODO: randomize size
        box_height = BOX_SIZE_REL * IMAGE_WIDTH
        box_width = 3 * BOX_SIZE_REL * IMAGE_HEIGHT
        # Set starting points (between 10% and 70% of the image)
        x0 = width * (0.1 + random.random() * 0.6)
        y0 = height * (0.1 + random.random() * 0.6)
        return [ (x0,y0),
                 (x0 + box_width, y0),
                 (x0 + box_width, y0 + box_height),
                 (x0, y0 + box_height) ]

    '''
        getTransparencyLevel
            Return the transparency level to use for the box.
    '''
    def getTransparencyLevel(self):
        return OPACITY_MIN + random.random() * (OPACITY_MAX - OPACITY_MIN)

    '''
        addBlackBox
            Add a black box to the image (TODO: trying to avoid black box
            overlap).
        Input:
            - image: the base image
            - black_boxes: matrix of same size as the image with each pixel
                           equals to 1 if is in a black box
    '''
    def addBlackBox(self, image, black_boxes):
        transparency_level = self.getTransparencyLevel()
        transparency = int(transparency_level * 255)
        if CHANNELS == 1:
            bb = Image.new('LA', image.size, (255, 0))
            box_color = (0, transparency)
        else:
            bb = Image.new('RGBA', image.size, (255, 255, 255, 0))
            box_color = (0, 0, 0, transparency)
        draw = ImageDraw.Draw(bb)
        draw.polygon(self.getRectangleVertices(image.size), fill=box_color)
        # Rotate the polygon
        angle = random.random() * 180
        bb = bb.rotate(angle, resample=Image.BILINEAR)
        # Paste onto the image
        image.paste(bb, (0,0), bb)
        black_boxes.paste(bb, (0,0), bb)
        return image, black_boxes


    '''
        modifyImage
            Add one or more black boxes of opacity in [OPACITY_MIN, OPACITY_MAX]
            in some random point of the image.
    '''
    def modifyImage(self, image):
        black_boxes = Image.new('RGBA', image.size, color=(255,255,255,0))
        # TODO: add more than 1 block?
        #for _ in range(random.randint(1,2)):
        image, black_boxes = self.addBlackBox(image, black_boxes)
        # Convert black_boxes to numpy array
        opacity = black_boxes.split()[-1]
        bb = standardizer_numpy(numpy.asarray(opacity) / 255.)
        # Reshape
        bb = bb.reshape(IMAGE_WIDTH*IMAGE_HEIGHT)
        return image, bb

    def flow(self, batch_size=BATCH_SIZE):
        x = []
        labels = []
        img_index = -1
        while 1:
            # Load a new image
            img_index = (img_index + 1) % len(self._files)
            path = self._files[img_index]
            rpath = os.path.join(self._folder, path)
            img = Image.open(rpath)
            # Split image
            for split_img in self._image_splitter.splitImage(
                                                    img,
                                                    self._split_size,
                                                    padding=False,
                                                    stride=self._stride_size):
                if len(labels) == batch_size:
                    x = numpy.asarray(x, dtype='float32')
                    labels = numpy.asarray(labels, dtype='float32')
                    yield (x, labels)
                    x = []
                    labels = []
                # Process split
                img_data, label = self.getInputData(split_img)
                x.append(img_data)
                labels.append(label)
