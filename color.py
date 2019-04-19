from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.applications import ResNet50
from keras.callbacks import Callback
import tensorflow as tf
import random
import glob
# import wandb
# from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
import cv2
from keras import backend as K
from skimage import io, color

# run = wandb.init(project='colorizer-applied-dl')
# = run.config

num_epochs = 1
batch_size = 4
img_dir = "images"
height = 256
width = 256

val_dir = 'test'
train_dir = 'train'

# automatically get the data if it doesn't exist
if not os.path.exists("train"):
    print("Downloading flower dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/l2kzone/flowers.tar | tar xz", shell=True)

def my_generator(batch_size, img_dir):
    """A generator that returns black and white images and color images"""
    image_filenames = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        bw_images = np.zeros((batch_size, width, height))
        color_images = np.zeros((batch_size, width, height, 3))
        random.shuffle(image_filenames) 
        if ((counter+1)*batch_size>=len(image_filenames)):
              counter = 0
        for i in range(batch_size):
              img = Image.open(image_filenames[counter + i]).resize((width, height))
              color_images[i] = np.array(img)
              bw_images[i] = np.array(img.convert('L'))
        yield (bw_images, color_images)
        counter += batch_size

def get_model(img_width, img_height):
      # reproduced from  Baldassarre et al [https://arxiv.org/pdf/1712.03400.pdf]
      # # ResNet...
      # res_net_input = Input(shape=(img_width, img_height, 3), name='resnet_input')
      # res_net = ResNet50(include_top=True, weights='imagenet')(res_net_input) # (None, 1000)
      
      image_input = Input(shape=(img_width, img_height, 1), name='input')
      # encoder
      enc1 = Conv2D(64, (3,3), activation='relu', strides=2, padding='same')(image_input)
      enc2 = Conv2D(128, (3,3), activation='relu', strides=1, padding='same')(enc1)
      enc3 = Conv2D(128, (3,3), activation='relu', strides=2, padding='same')(enc2)
      enc4 = Conv2D(256, (3,3), activation='relu', strides=1, padding='same')(enc3)
      enc5 = Conv2D(256, (3,3), activation='relu', strides=2, padding='same')(enc4)
      enc6 = Conv2D(512, (3,3), activation='relu', strides=1, padding='same')(enc5)
      enc7 = Conv2D(512, (3,3), activation='relu', strides=1, padding='same')(enc6)
      enc8 = Conv2D(256, (3,3), activation='relu', strides=1, padding='same')(enc7)

      # fusion
      # feature_replicated = tf.tile(feature_extractor, (W/8, W/8, 1))
      # fusion = concatenate((enc8, feature_replicated), axis=2)
      # fusion = Conv2D(256, (1,1), activation='relu', strides=1, padding='same')(fusion)

      # decoder
      dec1 = Conv2D(128, (3,3), activation='relu', strides=1, padding='same')(enc8)
      dec2 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(dec1)
      dec3 = Conv2D(64, (3,3), activation='relu', strides=1, padding='same')(dec2)
      dec4 = Conv2D(64, (3,3), activation='relu', strides=1, padding='same')(dec3)
      dec5 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(dec4)
      dec6 = Conv2D(32, (3,3), activation='relu', strides=1, padding='same')(dec5)
      dec7 = Conv2D(2, (3,3), activation='relu', strides=1, padding='same')(dec6)
      output = Conv2DTranspose(2, (2,2), strides=(2,2), padding='same')(dec7)

      model = Model(inputs=[image_input], outputs=[output])

      return model

      # model.add(Reshape((height,width,1), input_shape=(config.height,config.width)))
      # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
      # model.add(MaxPooling2D(2,2))
      # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
      # model.add(UpSampling2D((2, 2)))
      # model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

def perceptual_distance(y_true, y_pred):
    rmean = ( y_true[:,:,:,0] + y_pred[:,:,:,0] ) / 2
    r = y_true[:,:,:,0] - y_pred[:,:,:,0]
    g = y_true[:,:,:,1] - y_pred[:,:,:,1]
    b = y_true[:,:,:,2] - y_pred[:,:,:,2]
    
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


# (val_bw_images, val_color_images) = next(my_generator(145, val_dir))

# model.fit_generator( my_generator(batch_size, train_dir),
#                      steps_per_epoch=2,
#                      epochs=num_epochs, callbacks=[WandbCallback(data_type='image', predictions=16)],
#                      validation_data=(val_bw_images, val_color_images))
X = tf.zeros((1, 256, 256, 1))
Y = tf.zeros((1, 256, 256, 2))
model = get_model(256, 256)
learning_rate = 1e-5
model.compile(optimizer=Adam(lr=learning_rate), loss='mse')#, metrics=[perceptual_distance])
model.fit(X, Y, epochs=1, steps_per_epoch=1)
# res_net = ResNet50(include_top=True, weights='imagenet') # 1001x1x1
# res_net.compile(optimizer='adam', loss='categorical_crossentropy')
# res_net.fit(X, Y, epochs=1, steps_per_epoch=1)

