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

import tensorflow as tf


class CompleteTiledMaskInpaintings(object):
    """Creates complete image inpaintings from tiled masks.

    Attributes:
        divisor: int, the number of pixel blocks to skip until next masked
            pixel block. The integer reciprocal of the percent of masking.
        kernel_size: int, the number of pixels for each square mask block.
        batch_size: int, the number of images in a batch.
        height: int, the number of pixels for image height.
        width: int, the number of pixels for image width.
        depth: int, the numver of channels for images.
        tiled_boolean_mask: tensor, boolean tensor of which pixels to mask
            of shape (batch_size, height, width, depth, divisor).
    """
    def __init__(
        self, divisor, kernel_size, batch_size, height, width, depth
    ):
        """Initializes `CompleteTiledMaskInpaintings` instance.

        Args:
            divisor: int, the number of pixel blocks to skip until next masked
                pixel block. The integer reciprocal of the percent of masking.
            kernel_size: int, the number of pixels for each square mask block.
            batch_size: int, the number of images in a batch.
            height: int, the number of pixels for image height.
            width: int, the number of pixels for image width.
            depth: int, the numver of channels for images.
        """
        self.divisor = divisor
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth = depth
        self.tiled_boolean_mask = self.create_boolean_mask()

    def create_boolean_mask(self):
        """Creates boolean mask of which pixels to mask for each tesselation.

        Returns:
            tiled_boolean_mask: tensor, boolean tensor of which pixels to mask
                of shape (batch_size, height, width, depth, divisor).
        """
        reduced_height = (self.height // self.kernel_size)
        reduced_width = (self.width // self.kernel_size)
        num_pixels = reduced_height * reduced_width
        # shape = (reduced_height * reduced_width).
        indices = tf.range(start=0, limit=num_pixels, delta=1, dtype=tf.int32)
        # shape = (1, reduced_height, reduced_width, 1).
        indices_image = tf.reshape(
            tensor=indices, shape=(1, reduced_height, reduced_width, 1)
        )
        # shape = (1, reduced_height, reduced_width, 1).
        modded_indices_image = indices_image % self.divisor

        # shape = (height, width, 1).
        resized_modded_indices_image = tf.squeeze(
            input=tf.image.resize(
                images=modded_indices_image,
                size=(self.height, self.width),
                method="nearest"
            ),
            axis=0
        )

        # shape = (height, width, divisor).
        tiled_modded_indices_image = tf.tile(
            input=resized_modded_indices_image, multiples=(1, 1, self.divisor)
        )

        # shape = (divisor,).
        mod_range = tf.range(
            start=0, limit=self.divisor, delta=1, dtype=tf.int32
        )

        # shape = (height, width, divisor).
        boolean_mask = tf.equal(x=tiled_modded_indices_image, y=mod_range)
        # shape = (1, height, width, 1, divisor).
        expanded_boolean_mask = tf.reshape(
            tensor=boolean_mask,
            shape=(1, self.height, self.width, 1, self.divisor)
        )
        # shape = (batch_size, height, width, depth, divisor).
        tiled_boolean_mask = tf.tile(
            input=expanded_boolean_mask,
            multiples=(self.batch_size, 1, 1, self.depth, 1)
        )

        return tiled_boolean_mask

    def create_masked_images(self, images):
        """Creates masked images where every pixel will eventually be masked.

        Args:
            images: tensor, real images from dataset of shape
                (batch_size, height, width, depth).

        Returns:
            concat_unstacked_masked_images: tensor, masked images of shape
                (batch_size * divisor, height, width, depth).
        """
        # shape = (batch_size, height, width, depth, divisor).
        tiled_images = tf.tile(
            input=tf.expand_dims(input=images, axis=-1),
            multiples=(1, 1, 1, 1, self.divisor)
        )
        # shape = (batch_size, height, width, depth, divisor).
        masked_images = tf.where(
            condition=self.tiled_boolean_mask,
            x=-tf.ones_like(input=tiled_images, dtype=tf.float32),
            y=tiled_images
        )
        # list len = divisor, shape = (batch_size, height, width, depth).
        unstacked_masked_images = tf.unstack(value=masked_images, axis=-1)
        # shape = (batch_size * divisor, height, width, depth).
        concat_unstacked_masked_images = tf.concat(
            values=unstacked_masked_images, axis=0
        )

        return concat_unstacked_masked_images

    def gather_masked_image_inpaintings(self, images):
        """Gathers masked image inpaintings to form full inpainted image.

        Args:
            images: tensor, inpainted images from generator of shape
                (batch_size, height, width, depth).

        Returns:
            masked_images_inpaintings: tensor, fully inpainted images of shape
                (batch_size, height, width, depth).
        """
        # list len = divisor, shape = (batch_size, height, width, depth).
        split_images = tf.split(
            value=images, num_or_size_splits=self.divisor, axis=0
        )
        # shape = (batch_size, height, width, depth, divisor).
        stacked_split_images = tf.stack(values=split_images, axis=-1)
        # shape = (batch_size * height * width * depth).
        flat_masked_images_inpaintings = tf.boolean_mask(
            tensor=tf.reshape(stacked_split_images, shape=-1),
            mask=tf.reshape(self.tiled_boolean_mask, shape=-1),
            axis=0
        )

        # shape = (batch_size, height, width, depth).
        masked_images_inpaintings = tf.reshape(
            tensor=flat_masked_images_inpaintings,
            shape=(-1, images.shape[1], images.shape[2], images.shape[3])
        )

        return masked_images_inpaintings
