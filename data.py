import os
import numpy as np
from skimage import io, color
from skimage.transform import resize

data_path = 'data/'
image_rows = 480
image_cols = 500

def create_data(dataset='train'):
    train_path = os.path.join(data_path, dataset)
    images = os.listdir(train_path)
    total = len(images)

    L = np.zeros((total, image_rows, image_cols, 1))
    ab = np.zeros((total, image_rows, image_cols, 2))

    print('-'* 30)
    print('Creating ' +dataset+ ' images...')
    print('-' * 30)
    for i, image_name in enumerate(images):
        img = io.imread(os.path.join(train_path, image_name))
        lab = resize(color.rgb2lab(img), (image_rows, image_cols, 3))
        L[i] = np.array(lab[:,:,0:1])
        ab[i] = np.array(lab[:,:,1:])

    np.save('imgs_'+dataset+'_L.npy', L)
    np.save('imgs_'+dataset+'_ab.npy', ab)
    print("Saved .npy files")

def load_data(dataset='train'):
    L = np.load('imgs_'+dataset+'_L.npy')
    ab = np.load('imgs_'+dataset+'_ab.npy')
    return L, ab

if __name__ == '__main__':
    create_data('train')
    create_data('test')



