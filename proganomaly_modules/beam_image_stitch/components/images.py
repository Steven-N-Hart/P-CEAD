# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import

import argparse
from collections import defaultdict
import datetime
import json
import logging
import math
import os
import re


import apache_beam as beam
from apache_beam.io.gcp import gcsio
import cv2
import matplotlib.pyplot as plt
import openslide
import numpy as np
import skimage.color
import skimage.filters
import tensorflow as tf


class LeafCombineDoFn(beam.DoFn):
    """ParDo class that combines leaf images of 4-ary tree.

    Attributes:
        stitch_type: str, type of image being stitched.
    """
    def __init__(self, stitch_type):
        """Constructor of ParDo class that combines leaf images of 4-ary tree.

        Args:
            stitch_type: str, type of image being stitched.
        """
        self.stitch_type = stitch_type

    def process(self, inference_dict):
        """Combines leaf images of 4-ary tree.

        Args:
            inference_dict: dict, dictionary of images & index stacks. 

        Returns:
            List of 2-tuple of grid index and dictionary of images & index stacks.
        """
        combined_dict = {}
        grid_idx = inference_dict["grid_global_idx_stack"][-1]
        inner_idx = inference_dict["grid_local_idx_stack"][-1]
        inner_dict = {
            "image": inference_dict["images"][self.stitch_type],
            "grid_global_idx_stack": inference_dict["grid_global_idx_stack"][:-1],
            "grid_local_idx_stack": inference_dict["grid_local_idx_stack"][:-1]
        }
        combined_dict[inner_idx] = inner_dict
        return [(grid_idx, combined_dict)]


class BranchCombineDoFn(beam.DoFn):
    """ParDo class that combines branch images of 4-ary tree.

    Attributes:
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
    """
    def __init__(self, patch_height, patch_width):
        """Constructor of ParDo class that combines branch images of 4-ary tree.

        Args:
            patch_height: int, the height in pixels of an image patch.
            patch_width: int, the width in pixels of an image patch.
        """
        self.patch_height = patch_height
        self.patch_width = patch_width

    def merge_images_into_square(
        self, top_left, top_right, bottom_left, bottom_right):
        """Combines leaf images of 4-ary tree.

        Args:
            top_left: tensor, top left image of 4-square of images.
            top_right: tensor, top right image of 4-square of images.
            bottom_left: tensor, bottom left image of 4-square of images.
            bottom_right: tensor, bottom right image of 4-square of images.

        Returns:
            Combined and resized 4-square image tensor.
        """
        top_row = tf.concat(values=[top_left, top_right], axis=1)
        bottom_row = tf.concat(values=[bottom_left, bottom_right], axis=1)
        square = tf.concat(values=[top_row, bottom_row], axis=0)
        resized = tf.squeeze(
            tf.image.resize(
                images=tf.expand_dims(input=square, axis=0),
                size=(self.patch_height, self.patch_width)
            ),
            axis=0
        )
        return resized

    def process(self, my_tuple):
        """Combines leaf images of 4-ary tree.

        Args:
            my_tuple: 2-tuple, grid index and dictionary of images,
                segmentation coordinates, and index stacks.

        Returns:
            List of 2-tuple of grid index and dictionary of images,
                segmentation coordinates, and index stacks.
        """
        prior_grid_idx, branch_dict_list = my_tuple
        branch_dict = {k: v for d in branch_dict_list for k, v in d.items()}
        combined_dict = {}
        if branch_dict[0]["grid_global_idx_stack"]:
            grid_idx = branch_dict[0]["grid_global_idx_stack"].pop()
            inner_idx = branch_dict[0]["grid_local_idx_stack"].pop()
        else:
            grid_idx = 0
            inner_idx= 0
        merged_image = self.merge_images_into_square(
            top_left=branch_dict[0]["image"],
            top_right=branch_dict[1]["image"],
            bottom_left=branch_dict[2]["image"],
            bottom_right=branch_dict[3]["image"]
        )
        combined_dict[inner_idx] = {
            "image": merged_image,
            "grid_global_idx_stack": branch_dict[0]["grid_global_idx_stack"],
            "grid_local_idx_stack": branch_dict[0]["grid_local_idx_stack"]
        }
        return [(grid_idx, combined_dict)]


class WriteImageDoFn(beam.DoFn):
    """ParDo class that writes fully stitched images to GCS.

    Attributes:
        output_filename: str, the output filename in GCS.
    """
    def __init__(self, output_filename):
        """Constructor of ParDo class that writes fully stitched images to GCS.

        Args:
            output_filename: str, the output filename in GCS.
        """
        self.output_filename = output_filename + ".png"

    def descale_images(self, images):
        """Descales images from [-1., 1.] to [0, 255].

        Args:
            images: np.array, array of images with range [-1., 1.] of shape
                (num_images, height, width, num_channels).
        Returns:
            Tensor of images with range [0, 255] of shape
                (num_images, height, width, num_channels).
        """
        image = tf.cast(
            x=((images + 1.0) * (255. / 2.)), dtype=tf.uint8
        )

        image = tf.where(
            condition=image < 0,
            x=tf.zeros_like(input=image, dtype=tf.uint8),
            y=image
        )
        image = tf.where(
            condition=image > 255,
            x=tf.ones_like(input=image, dtype=tf.uint8) * 255,
            y=image
        )
        return image

    def process(self, my_tuple):
        """Writes fully stitched images to GCS.

        Args:
            my_tuple: 2-tuple, grid index and dictionary of images & index
                stacks.

        Returns:
            Empty list collection.
        """
        prior_grid_idx, branch_dict = my_tuple
        image = branch_dict[0]["image"]
        image = self.descale_images(images=image)
        png = tf.image.encode_png(image)
        tf.io.write_file(filename=self.output_filename, contents=png)
        return []
