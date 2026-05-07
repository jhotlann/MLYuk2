import tensorflow as tf
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import os
import numpy as np
from PIL import Image
import glob

def build_encoder():
    model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    return model


def extract_feature(img_paths, model, batch_size):
    features = {}
    batches = [img_paths[i:i+batch_size] for i in range(0, len(img_paths), batch_size)]

    for batch_paths in batches:
        imgs = load_batch(batch_paths)
        res = model.predict(imgs)

        for path, vec in zip(batch_paths, res):
            file_name = os.path.basename(path)
            features[file_name] = res

    return features


def load_image(path, image_size=(299,299)):
    img = Image.open(path)
    img = img.resize(image_size)
    arr = np.array(img)
    arr = preprocess_input(arr)
    return arr

def load_batch(paths, image_size=(299,299)):
    return np.stack([load_image(path, image_size) for path in paths])

def save_feature():
    encoder = build_encoder()
    flickr_paths = glob.glob('data/flickr8k/Images/*.jpg')
    features = extract_feature(flickr_paths, encoder, 32)
    np.save('features/flickr8k_features.npy', features)
    