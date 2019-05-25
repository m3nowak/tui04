import contextlib
import csv
import os
import pickle

import numpy as np
import skimage
from skimage.feature import hog

IMAGE_COUNT = 10000
FEATURE_SIZE = 40500

FEATURE_FILE_CSV = 'artifacts/faces-celeba/list_attr_celeba.csv'
IMAGES_FOLDER = 'artifacts/faces-celeba/images'
DUMP_LOCATION = 'artifacts/faces-celeba/learning_model.pkl'

def create_dataset():
    image_counter = 0
    image_hog_array = np.zeros((IMAGE_COUNT,FEATURE_SIZE))
    feature_list = []
    with contextlib.closing(open(FEATURE_FILE_CSV, newline='')) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if image_counter % 100 == 0:
                print("{}/{}".format(image_counter, IMAGE_COUNT))
            is_male = 1 if row['Male'] == '1' else 0
            image = skimage.io.imread(os.path.join(IMAGES_FOLDER, row['image_id']), as_gray=True)
            image_hog = hog(image)
            image_hog_array[image_counter] = image_hog
            feature_list.append(is_male)
            image_counter += 1
            if image_counter >= IMAGE_COUNT:
                break
    feature_array = np.array(feature_list)
    return image_hog_array, feature_array

def dump_dataset(image_hog_array, feature_array):
    with contextlib.closing(open(DUMP_LOCATION, 'wb')) as pfile:
        pickle.dump((image_hog_array, feature_array), pfile)

def load_dataset():
    with contextlib.closing(open(DUMP_LOCATION, 'rb')) as pfile:
        (image_hog_array, feature_array) = pickle.load(pfile)
    return image_hog_array, feature_array