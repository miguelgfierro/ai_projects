#This file converts an image dataset to grayscale
#Ex: python convert_image_dataset_to_grayscale.py --root_folder /datadrive/food101/food-101/images --dest_folder /datadrive/food101/food-101/images_grayscale --verbose

import time
import argparse
from utils import convert_image_dataset_to_grayscale


parser = argparse.ArgumentParser(description='This file converts an image dataset to grayscale')
parser.add_argument('--root_folder', type=str, help='root folder of dataset')
parser.add_argument('--dest_folder', type=str, help='destination folder to save grayscale images')
parser.add_argument('--verbose', action='store_true', help='verbose')
args = parser.parse_args()

since = time.time()
convert_image_dataset_to_grayscale(args.root_folder, args.dest_folder, verbose=args.verbose)
print("Process finished in {}s".format(time.time() - since))

