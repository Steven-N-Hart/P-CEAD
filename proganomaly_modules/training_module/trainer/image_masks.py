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


def calculate_image_mask_block_sizes_per_resolution(
    num_resolutions, min_height, min_width, pixel_mask_percent
):
    """Calculates image mask block sizes for each image resolution.

    Args:
        num_resolutions: int, number of resolutions for progressive growing.
        min_height: int, minimum height of first progressive growth.
        min_width: int, minimum width of first progressive growth.
        pixel_mask_percent: float, percent of pixels to mask within image.

    Returns:
        image_mask_block_sizes: list, list of list of ints describing the mask
            block sizes.
    """
    image_mask_block_sizes = []
    for i in range(num_resolutions):
        height = min_height * 2 ** i
        width = min_width * 2 ** i
        num_pixels = height * width

        num_masked_pixels = int(num_pixels * pixel_mask_percent)

        block_sizes = []
        while num_masked_pixels > 0:
            block_size = num_masked_pixels // 2
            block_size = min(max(1, block_size), num_masked_pixels)
            block_sizes.append(block_size)
            num_masked_pixels -= block_size
        image_mask_block_sizes.append(block_sizes)

    return image_mask_block_sizes


class ImageMasks(object):
    """Class used for masking images.

    Attributes:
        params: dict, user passed parameters.
    """
    def __init__(self):
        """Instantiate instance of `ImageMasks`.
        """
        pass

    def create_mask_block_corners_1d(self, size, block_lengths):
        """Creates corner coordinates for 1D for each block for each image.

        Args:
            size: int, the max length of the dimension.
            block_lengths: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the lengths of each
                block for each image.

        Returns:
            start_corners_good: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D coordinates of
                the starting corner of each block for each image.
            end_corners_good: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D coordinates of
                the ending corner of each block for each image.
        """
        # shape = (batch_size, num_blocks).
        start_corners_bad = tf.random.uniform(
            shape=tf.shape(input=block_lengths),
            minval=0,
            maxval=size,
            dtype=tf.int32
        )

        # shape = (batch_size, num_blocks).
        end_corners_bad = block_lengths + start_corners_bad

        # shape = (batch_size, num_blocks).
        end_overage = size - end_corners_bad

        # shape = (batch_size, num_blocks).
        corner_offset_fix = tf.where(
            condition=tf.less(x=end_overage, y=0),
            x=end_overage,
            y=tf.zeros_like(input=block_lengths)
        )

        # shape = (batch_size, num_blocks).
        start_corners_good = start_corners_bad + corner_offset_fix

        # shape = (batch_size, num_blocks).
        end_corners_good = block_lengths + start_corners_good

        return start_corners_good, end_corners_good

    def create_mask_block_corners(
        self, batch_size, height, width, growth_idx
    ):
        """Generates mask given image size and mask block corner coordinates.

        Args:
            batch_size: int, the number of elements in the minibatch.
            height: int, the height of images.
            width: int, the height of images.
            growth_idx: int, the current resolution growth index.

        Returns:
            x_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the starting corner of each block for each image.
            y_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the starting corner of each block for each image.
            x_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the ending corner of each block for each image.
            y_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the ending corner of each block for each image.
        """
        generator_params = self.params["generator"]
        ganomaly_params = generator_params["GANomaly"]
        block_idx = (growth_idx + 1) // 2

        # shape = (num_blocks,).
        mask_block_sizes = tf.convert_to_tensor(
            value=ganomaly_params["image_mask_block_sizes"][block_idx],
            dtype=tf.int32
        )

        # shape = (batch_size, num_blocks).
        tiled_mask_block_sizes = tf.tile(
            input=tf.expand_dims(input=mask_block_sizes, axis=0),
            multiples=[batch_size, 1]
        )

        # shape = (batch_size, num_blocks).
        if (generator_params["architecture"] == "GANomaly" and
            ganomaly_params["image_mask_block_random_shift_amount"] > 0):
            random_shifts = tf.random.uniform(
                shape=tf.shape(input=tiled_mask_block_sizes),
                minval=-(
                    ganomaly_params["image_mask_block_random_shift_amount"]
                ),
                maxval=(
                    ganomaly_params["image_mask_block_random_shift_amount"]
                ),
                dtype=tf.int32
            )
        else:
            random_shifts = tf.zeros_like(input=tiled_mask_block_sizes)

        # shape = (batch_size, num_blocks).
        random_shifted_block_sizes = tiled_mask_block_sizes + random_shifts

        # shape = (batch_size, num_blocks).
        mask_block_heights = tf.math.maximum(
            x=tf.cast(
                x=tf.sqrt(
                    x=tf.cast(x=random_shifted_block_sizes, dtype=tf.float32)
                ),
                dtype=tf.int32
            ),
            y=1
        )

        # shape = (batch_size, num_blocks).
        mask_block_widths = random_shifted_block_sizes // mask_block_heights

        # shape = (batch_size, num_blocks).
        x_start_corners, x_end_corners = self.create_mask_block_corners_1d(
            size=width, block_lengths=mask_block_widths
        )

        # shape = (batch_size, num_blocks).
        y_start_corners, y_end_corners = self.create_mask_block_corners_1d(
            size=height, block_lengths=mask_block_heights
        )

        return (
            x_start_corners,
            y_start_corners,
            x_end_corners,
            y_end_corners
        )

    def generate_mask(self, height, width, x1, y1, x2, y2):
        """Generates mask given image size and mask block corner coordinates.

        Args:
            height: int, the height of images.
            width: int, the height of images.
            x1: tensor, rank 0 tensor of mask block's starting corner x
                coordinate.
            y1: tensor, rank 0 tensor of mask block's starting corner y
                coordinate.
            x2: tensor, rank 0 tensor of mask block's ending corner x
                coordinate.
            y2: tensor, rank 0 tensor of mask block's ending corner y
                coordinate.

        Returns:
            mask: tensor, rank 2 tensor of shape (height, width) containing
                mask block.
        """
        # shape = (width, y1).
        outer_first_block = tf.zeros(shape=(width, y1), dtype=tf.int32)

        inner_first_block = tf.zeros(shape=(x1, y2 - y1), dtype=tf.int32)
        inner_second_block = tf.ones(shape=(x2 - x1, y2 - y1), dtype=tf.int32)
        inner_third_block = tf.zeros(
            shape=(width - x2, y2 - y1), dtype=tf.int32
        )

        # shape = (width, y2 - y1).
        outer_second_block = tf.concat(
            values=[inner_first_block, inner_second_block, inner_third_block],
            axis=0
        )

        # shape = (width, height - y2).
        outer_third_block = tf.zeros(
            shape=(width, height - y2), dtype=tf.int32
        )

        # shape = (height, width).
        mask = tf.concat(
            values=[outer_first_block, outer_second_block, outer_third_block],
            axis=1
        )

        return mask

    def create_image_masks(
        self,
        batch_size,
        height,
        width,
        depth,
        x_start_corners,
        y_start_corners,
        x_end_corners,
        y_end_corners
    ):
        """Creates image masks.

        Args:
            batch_size: int, the number of elements in the minibatch.
            height: int, the height of images.
            width: int, the width of images.
            depth: int, the number of channels of images.
            x_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the starting corner of each block for each image.
            y_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the starting corner of each block for each image.
            x_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the ending corner of each block for each image.
            y_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the ending corner of each block for each image.

        Returns:
            pixel_masks: tensor, rank 3 tensor of pixel masks of shape
                (batch_size, height, width).
        """
        # shape = (batch_size, height, width).
        pixel_masks = tf.map_fn(
            # shape = (height, width).
            fn=lambda batch_idx: tf.reduce_sum(
                # shape = (num_blocks, height, width).
                input_tensor=tf.map_fn(
                    fn=lambda x: self.generate_mask(
                        height, width, x1=x[0], y1=x[1], x2=x[2], y2=x[3]
                    ),
                    elems=(
                        x_start_corners[batch_idx],
                        y_start_corners[batch_idx],
                        x_end_corners[batch_idx],
                        y_end_corners[batch_idx]
                    ),
                    fn_output_signature=tf.int32
                ),
                axis=0
            ),
            elems=tf.range(start=0, limit=batch_size, delta=1, dtype=tf.int32),
            fn_output_signature=tf.int32
        )

        # shape = (batch_size, height, width, depth).
        image_masks = tf.tile(
            input=tf.expand_dims(input=pixel_masks, axis=-1),
            multiples=(1, 1, 1, depth)
        )

        return image_masks

    def shuffle_mask_block_pixels(
        self, image, mask, block_height, block_width
    ):
        """Shuffles pixels within mask block.

        Args:
            image: tensor, rank 3 image tensor of shape
                (height, width, depth).
            mask: tensor, rank 2 tensor of shape (height, width) containing
                mask block.
            block_height: tensor, rank 0 tensor of mask block's height.
            block_width: tensor, rank 0 tensor of mask block's width.

        Returns:
            shuffled_block_image: tensor, rank 3 image tensor of shape
                (height, width, depth) with mask block pixels shuffled within
                zero image.

        """
        # shape = (height, width, depth).
        tiled_mask = tf.equal(
            x=tf.tile(
                input=tf.expand_dims(input=mask, axis=-1),
                multiples=(1, 1, image.shape[-1])
            ),
            y=1
        )

        # shape = (block_height * block_width * depth).
        mask_block_flat = tf.boolean_mask(
            tensor=image, mask=tiled_mask
        )

        # shape = ((height - block_height) * (width - block_width) * depth).
        remaining_pixels_flat = tf.boolean_mask(
            tensor=image, mask=~tiled_mask
        )

        # shape = (block_height * block_width, depth).
        block_pixels = tf.reshape(
            tensor=mask_block_flat,
            shape=(-1, image.shape[-1])
        )

        # shape = (block_height * block_width, depth).
        shuffled_pixels = tf.random.shuffle(value=block_pixels)

        # shape = (block_height * block_width, depth).
        shuffled_pixels_flat = tf.reshape(tensor=shuffled_pixels, shape=(-1,))

        # shape = (block_height * block_width * depth, depth).
        block_indices = tf.where(condition=tiled_mask)

        # shape = (height, width, depth).
        shuffled_block_image = tf.scatter_nd(
            indices=tf.cast(x=block_indices, dtype=tf.int32),
            updates=shuffled_pixels_flat,
            shape=tf.shape(image)
        )

        return shuffled_block_image

    def create_shuffle_masks(
        self,
        batch_size,
        height,
        width,
        x_start_corners,
        y_start_corners,
        x_end_corners,
        y_end_corners,
        images
    ):
        """Creates shuffle masks.

        Args:
            batch_size: int, the number of elements in the minibatch.
            height: int, the height of images.
            width: int, the height of images.
            x_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the starting corner of each block for each image.
            y_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the starting corner of each block for each image.
            x_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the ending corner of each block for each image.
            y_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the ending corner of each block for each image.
            images: tensor, rank 4 image tensor of shape
                (batch_size, height, width, depth).

        Returns:
            shuffle_masks: tensor, rank 4 tensor of pixel masks of shape
                (batch_size, height, width, depth).
        """
        # shape = (batch_size, height, width, depth).
        shuffle_masks = tf.map_fn(
            # shape = (height, width, depth).
            fn=lambda batch_idx: tf.reduce_sum(
                # shape = (num_blocks, height, width, depth).
                input_tensor=tf.map_fn(
                    fn=lambda x: self.shuffle_mask_block_pixels(
                        image=images[batch_idx],
                        mask=self.generate_mask(
                            height, width, x1=x[0], y1=x[1], x2=x[2], y2=x[3]
                        ),
                        block_height=x[3] - x[1],
                        block_width=x[2] - x[0]
                    ),
                    elems=(
                        x_start_corners[batch_idx],
                        y_start_corners[batch_idx],
                        x_end_corners[batch_idx],
                        y_end_corners[batch_idx]
                    ),
                    fn_output_signature=tf.float32
                ),
                axis=0
            ),
            elems=tf.range(start=0, limit=batch_size, delta=1, dtype=tf.int32),
            fn_output_signature=tf.float32
        )

        return shuffle_masks

    def shuffle_mask_images(
        self,
        batch_size,
        height,
        width,
        x_start_corners,
        y_start_corners,
        x_end_corners,
        y_end_corners,
        images,
        image_masks
    ):
        """Masks images with irregular shapes of shuffled pixels.

        Args:
            batch_size: int, the number of elements in the minibatch.
            height: int, the height of images.
            width: int, the height of images.
            x_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the starting corner of each block for each image.
            y_start_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the starting corner of each block for each image.
            x_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D x coordinates of
                the ending corner of each block for each image.
            y_end_corners: tensor, rank 2 tensor of shape
                (batch_size, num_blocks) which contain the 1D y coordinates of
                the ending corner of each block for each image.
            images: tensor, rank 4 image tensor of shape
                (batch_size, height, width, depth).
            image_masks: tensor, rank 4 integer mask tensor of shape
                (batch_size, height, width, depth).

        Returns:
            masked_images: tensor, rank 4 masked image tensor of shape
                (batch_size, height, width, depth).
        """
        shuffle_masks = self.create_shuffle_masks(
            batch_size,
            height,
            width,
            x_start_corners,
            y_start_corners,
            x_end_corners,
            y_end_corners,
            images
        )

        # shape = (batch_size, height, width, depth).
        masked_images = tf.where(
            condition=tf.equal(x=image_masks, y=1),
            x=shuffle_masks,
            y=images
        )

        return masked_images

    def dead_mask_images(self, images, image_masks):
        """Masks images with irregular shapes of dead pixels.

        Args:
            images: tensor, rank 4 image tensor of shape
                (batch_size, height, width, depth).
            image_masks: tensor, rank 4 integer mask tensor of shape
                (batch_size, height, width, depth).

        Returns:
            masked_images: tensor, rank 4 masked image tensor of shape
                (batch_size, height, width, depth).
        """
        # shape = (batch_size, height, width, depth).
        masked_images = tf.where(
            condition=tf.equal(x=image_masks, y=1),
            x=-tf.ones_like(input=images),
            y=images
        )

        return masked_images

    def mask_images(self, images, growth_idx):
        """Masks images with irregular shapes.

        Args:
            images: tensor, rank 4 image tensor of shape
                (batch_size, height, width, depth).
            growth_idx: int, the current resolution growth index.

        Returns:
            masked_images: tensor, rank 4 masked image tensor of shape
                (batch_size, height, width, depth).
        """
        batch_size = tf.shape(input=images)[0]
        height, width, depth = images.shape[1:]

        # each shape = (batch_size, num_blocks).
        (
            x_start_corners,
            y_start_corners,
            x_end_corners,
            y_end_corners
        ) = self.create_mask_block_corners(
            batch_size, height, width, growth_idx
        )

        # shape = (batch_size, height, width, depth).
        image_masks = self.create_image_masks(
            batch_size,
            height,
            width,
            depth,
            x_start_corners,
            y_start_corners,
            x_end_corners,
            y_end_corners
        )

        # shape = (batch_size, height, width, depth).
        if (self.params["generator"]["architecture"] == "GANomaly" and
            self.params["generator"]["GANomaly"]["use_shuffle_image_masks"]):
            masked_images = self.shuffle_mask_images(
                batch_size,
                height,
                width,
                x_start_corners,
                y_start_corners,
                x_end_corners,
                y_end_corners,
                images,
                image_masks
            )
        else:
            masked_images = self.dead_mask_images(images, image_masks)

        return masked_images
