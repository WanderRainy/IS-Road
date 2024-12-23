#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from tqdm import tqdm
from prepare_data import pycococreatortools
from numba import jit
from multiprocessing.pool import Pool

ROOT_DIR = 'city_scale/20cities_patch/Instance_road/validate'
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "Instance Road Dataset City_Scale",
    "url": "**********",
    "version": "0.1.0",
    "year": 2023,
    "contributor": "Ruoyu Yang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'road',
        'supercategory': 'road',
    },
    # {
    #     'id': 2,
    #     'name': 'circle',
    #     'supercategory': 'shape',
    # },
    # {
    #     'id': 3,
    #     'name': 'triangle',
    #     'supercategory': 'shape',
    # },
]

# @jit
def filter_for_jpeg(root, files):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

# @jit
def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
# @jit
def image2coco(image_filename,image_id, segmentation_id):
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(
        image_id, os.path.basename(image_filename), image.size)
    coco_output["images"].append(image_info)

    # filter for associated png annotations
    for root, _, files in os.walk(ANNOTATION_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)

        # go through each associated annotation
        for annotation_filename in annotation_files:

            # print(annotation_filename)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in os.path.basename(annotation_filename)][0]

            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(Image.open(annotation_filename)
                                     .convert('1')).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1

    image_id = image_id + 1
    return image_id, segmentation_id


if __name__ == "__main__":
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in tqdm(image_files):
            image_id, segmentation_id = image2coco(image_filename,image_id, segmentation_id)

    with open('{}/instances_road_validate.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
