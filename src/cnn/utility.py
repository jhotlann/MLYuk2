import numpy as np
from PIL import Image
import keras
from keras.applications import ResNet50
from keras.preprocessing import image as keras_image
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import os
import hashlib

def image_loader(path, h=224, w=224):
    im = Image.open(path)

    if h != None and w != None:
        im = im.resize((w, h))

    im_arr = np.array(im)
    norm_im = (im_arr  - np.min(im_arr))/(np.max(im_arr) - np.min(im_arr))

    return norm_im

def batch_loader(paths, h=224, w=224, c=3):
    batch_array = np.zeros((len(paths), h, w, c), dtype=np.float32)
    
    for idx, path in enumerate(paths):
        img = image_loader(path, h, w)

        if c == 1 and len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        elif c == 3 and len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        batch_array[idx] = img[:h, :w, :c]

    return batch_array


def feature_extractor(image_paths, model_name="ResNet50", cache_dir="./cache", img_size=(224, 224), use_cache=True):

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    paths_str = '|'.join(sorted(image_paths))
    paths_hash = hashlib.md5(paths_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"features_{model_name}_{paths_hash}.npy")
    
    if use_cache and os.path.exists(cache_file):
        print(f" Load features dari: {cache_file}")
        return np.load(cache_file)
    
    print(f"Loading model {model_name}...")
    base_model = ResNet50(weights='imagenet', include_top=False, 
                          input_shape=(img_size[0], img_size[1], 3))
    
    # Freeze weights
    base_model.trainable = False
    
    feature_model = Model(inputs=base_model.input, 
                         outputs=base_model.output)
    
    features_list = []
    
    for idx, path in enumerate(image_paths):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(image_paths)}")
        
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((img_size[1], img_size[0]))
            
            # preprocess
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)        
            img_array = preprocess_input(img_array)
            
            # Extract feature
            feature_vector = feature_model.predict(img_array, verbose=0)
            feature_vector = feature_vector.reshape(feature_vector.shape[0], -1)[0]
            features_list.append(feature_vector)
            
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            if features_list:
                features_list.append(np.zeros_like(features_list[0]))
            else:
                features_list.append(np.zeros(2048))
    
    features_array = np.array(features_list, dtype=np.float32)

    np.save(cache_file, features_array)
    print(f"Shape: {features_array.shape}")
    return features_array
