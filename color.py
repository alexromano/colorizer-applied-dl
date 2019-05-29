from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from keras.datasets import mnist
# from keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import random
# import glob
import math
# import wandb
# from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
import cv2
from keras import backend as K
from skimage import io, color
from skimage.transform import resize
from data import load_data

# run = wandb.init(project='colorizer-applied-dl')
# = run.config

num_epochs = 50
batch_size = 8
img_dir = "images"
height = 256
width = 256
original_width = 500
original_height = 480

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data/train"):
    print("Downloading flower dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/l2kzone/flowers.tar | tar xz --directory data", shell=True)

train_size = len(os.listdir('data/train'))
test_size = len(os.listdir('data/train'))

def get_model(img_width, img_height, learning_rate):
      # reproduced from  Baldassarre et al [https://arxiv.org/pdf/1712.03400.pdf]
      # # ResNet...
      # res_net_input = Input(shape=(img_width, img_height, 3), name='resnet_input')
      # res_net = ResNet50(include_top=True, weights='imagenet')(res_net_input) # (None, 1000)
      # u-net style
      input_images = Input(shape=(width, height, 1), name='input')
      conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_images)
      conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
      pool1 = MaxPooling2D(2)(conv1)

      conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
      conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
      pool2 = MaxPooling2D(2)(conv2)

      conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
      conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
      pool3 = MaxPooling2D(2)(conv3)

      conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
      conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv4)
      pool4 = MaxPooling2D(2)(conv4)

      conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool4)
      conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(conv5)

      conv6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv5), conv4])
      # conv6 = SpatialDropout2D(rate=0.5)(conv6)
      conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
      conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)

      conv7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6), conv3])
      # conv7 = SpatialDropout2D(rate=0.5)(conv7)
      conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
      conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)

      conv8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7), conv2])
      # conv8 = SpatialDropout2D(rate=0.5)(conv8)
      conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
      conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)

      conv9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8), conv1])
      # conv9 = SpatialDropout2D(rate=0.5)(conv9)
      conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
      conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)

      output = Conv2D(2, (1,1), activation='sigmoid')(conv9)

      model = Model(inputs=[input_images], outputs=[output])
      model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
      
      return model

def perceptual_distance(y_true, y_pred):
      rmean = ( y_true[:,:,:,0] + y_pred[:,:,:,0] ) / 2
      r = y_true[:,:,:,0] - y_pred[:,:,:,0]
      g = y_true[:,:,:,1] - y_pred[:,:,:,1]
      b = y_true[:,:,:,2] - y_pred[:,:,:,2]

      return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def preprocess(imgs):
      imgs_p = np.zeros((imgs.shape[0], height, width, imgs.shape[3]))
      for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], (height, width, imgs.shape[3]), preserve_range=True)
      return imgs_p


def train(learning_rate=1e-5):
      model = get_model(width, height, learning_rate)
      checkpoint = ModelCheckpoint('model.hd5')
      #input shape (None, 256, 256, 1) output shape (None, 256, 256, 2)
      L, ab = load_data('train')
      L, ab = preprocess(L), preprocess(ab)
      
      L_mean, L_std = np.mean(L), np.std(L)
      ab_mean, ab_std = np.mean(ab), np.std(ab)
      
      L -= L_mean
      L /= L_std
      ab -= ab_mean
      ab /= ab_std

      model.fit(L, ab, callbacks=[checkpoint])

def predict(batch_size):
      print('*'*30)
      print("Loading model")
      print('*'*30)
      model = load_model('model.hd5', compile=False)
      
      print('*'*30)
      print('Loading test data')
      print('*'*30)
      test_L, test_ab = load_data('test')
      test_L, test_ab = preprocess(test_L), preprocess(test_ab)
      test_size = len(test_L)

      lab = np.concatenate((test_L, test_ab), axis=3)
      test_rgb = np.array([color.lab2rgb(l) for l in lab])
      
      L_mean, L_std = np.mean(test_L), np.std(test_L)
      ab_mean, ab_std = np.mean(test_ab), np.std(test_ab)
      
      test_L -= L_mean
      test_L /= L_std
      test_ab -= ab_mean
      test_ab /= ab_std 
      
      print('*'*30)
      print('Running predictions')
      print('*'*30)
      predicted_ab = model.predict(test_L, batch_size=math.ceil(batch_size/test_size), verbose=1)

      rgb_predictions = np.zeros((predicted_ab.shape[0], height, width, 3))
      for i in range(predicted_ab.shape[0]):
            # concat original L with predicted ab, convert to rgb
            lab = np.concatenate((test_L[i], predicted_ab[i]), axis=2)
            rgb = np.array(color.lab2rgb(lab)) 
            rgb_predictions[i] = np.resize(rgb, (width, height, 3))
      # run perceptual distance
      return perceptual_distance(np.array(test_rgb), rgb_predictions)
