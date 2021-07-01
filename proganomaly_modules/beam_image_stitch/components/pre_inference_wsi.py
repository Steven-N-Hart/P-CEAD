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

from apache_beam.io.gcp import gcsio
import openslide
import math
import numpy as np
import skimage.color
import skimage.filters
import tensorflow as tf


def get_wsi_thumbnail(wsi_stitch_gcs_path, target_image_width):
    """Gets thumbnail from openslide WSI.

    Args:
        wsi_stitch_gcs_path: str, GCS path of WSI file.
        target_image_width: int, the approximate width of resultant thumbnail
            image.

    Returns:
        wsi: `OpenSlide` object of WSI.
        thumbnail: PIL.image, the thumbnail image from openslide.
    """
    gcs = gcsio.GcsIO()
    local_file = "slide_file.svs"

    wsi = None
    num_retries = 0
    while num_retries < 100:
        try:
            with open(local_file, "wb") as f:
                f.write(gcs.open(wsi_stitch_gcs_path).read())
            wsi = openslide.OpenSlide(filename=local_file)
        except:
            num_retries += 1
        else:
            break

    # Get the ratio for the target image width.
    divisor = int(wsi.level_dimensions[0][0] / target_image_width)
    # Get the height and width of the thumbnail using the ratio.
    patch_size_x = int(wsi.level_dimensions[0][0] / divisor)
    patch_size_y = int(wsi.level_dimensions[0][1] / divisor)
    # Extract the thumbnail.
    thumbnail = None
    num_retries = 0
    while num_retries < 100:
        try:
            thumbnail = wsi.get_thumbnail(size=(patch_size_x, patch_size_y))
        except:
            num_retries += 1
        else:
            break
    return wsi, thumbnail


def otsu_method(thumbnail):
    """Uses otsu method to create binary mask from thumbnail.

    Args:
        thumbnail: PIL.image, the thumbnail image from openslide.

    Returns:
        Binary np.array of shape (thumbnail_width, thumbnail_height, 1).
    """
    # Convert to grey scale image.
    gs_thumbnail = np.array(thumbnail.convert("L"))
    # Get the otsu threshold value.
    thresh = skimage.filters.threshold_otsu(image=gs_thumbnail)
    # Convert to binary mask.
    binary_img = gs_thumbnail < thresh
    binary_img = binary_img.astype(int)
    return binary_img


def rgb2hed_method(thumbnail, threshold):
    """Uses RGB2HED method to create binary mask from thumbnail.

    Args:
        thumbnail: PIL.image, the thumbnail image from openslide.
        threshold: float, threshold to convert image to binary mask.

    Returns:
        Binary np.array of shape (thumbnail_width, thumbnail_height, 1).
    """
    np_thumbnail = np.array(thumbnail)
    # Convert to hed space.
    hed_img = skimage.color.rgb2hed(rgb=np_thumbnail)

    # Convert to binary mask.
    binary_img = hed_img[:, :, 2] > threshold
    binary_img = binary_img.astype(int)
    return binary_img


def create_full_slide_mask(binary_img, slide_dims):
    """Creates full slide mask from binary image.

    Args:
        binary_img: np.array, binary array of shape
            (thumbnail_width, thumbnail_height, 1).
        slide_dims: 2-tuple of ints, (slide_width, slide_height) at level 0.

    Returns:
        Binary np.array of shape (slide_width, slide_height, 1).
    """
    binary_img_image = tf.reshape(
        tensor=binary_img,
        shape=(1, binary_img.shape[0], binary_img.shape[1], 1)
    )
    return tf.image.resize(
        images=binary_img_image,
        size=(slide_dims[1], slide_dims[0]),
        method="nearest"
    )[0, :, :, 0]


def wsi_build_grid(
    max_dim,
    slide_dims,
    patch_height,
    patch_width,
    full_slide_mask,
    include_patch_threshold,
    batch_size
):
    """Builds grid for 4-ary tree stitching of patches.

    Args:
        max_dim: int, the maximum dimension between height and width.
        slide_dims: 2-tuple of ints, (slide_width, slide_height) at level 0.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        full_slide_mask: np.array, binary array of shape
            (slide_width, slide_height, 1).
        include_patch_threshold: float, threshold to compare with percent of
            binary flags within a patch region to include in collection.
        batch_size: int, number of images to include in each batch for
            inference.

    Returns:
        List of dictionaries containing batch index, 4-ary tree indices and
            possibly coordinates and filename of patch PNG images to include
            in collection.
    """
    grid_list_of_lists_of_dicts = [
        [
            {}
            for j in range(max_dim)
        ]
        for i in range(max_dim)
    ]

    block_height = slide_dims[1] // patch_height
    block_width = slide_dims[0] // patch_width
    patches_added = 0
    for i in range(block_height):
        low_height = i * patch_height
        high_height = low_height + patch_height
        for j in range(block_width):
            low_width = j * patch_width
            high_width = low_width + patch_width
            counts = tf.reduce_sum(
                input_tensor=full_slide_mask[
                    low_height: high_height, low_width: high_width
                ]
            )
            percent = tf.cast(counts, tf.float32) / (patch_height * patch_width)
            if percent > include_patch_threshold:
                grid_list_of_lists_of_dicts[i][j]["coords"] = (
                    low_height, low_width
                )
                grid_list_of_lists_of_dicts[i][j]["batch_idx"] = (
                    patches_added // batch_size
                )
                patches_added += 1
        
    height, width = max_dim, max_dim
    depth = int(math.log(max_dim, 2))

    grid = [[[0] for _ in range(width)] for _ in range(height)]
    blocks = [
        [
            ((0, height), (0, width))
            for _ in range(width)
        ]
        for _ in range(height)
    ]
    indices = [[[0] * (depth + 1) for j in range(width)] for i in range(height)]
    factor_h = height
    factor_w = width
    for d in range(depth):
        factor_h = factor_h // 2
        factor_w = factor_w // 2
        if factor_h <= 0 or factor_w <= 0:
            break
        for i in range(height):
            for j in range(width):
                depth_start_h = blocks[i][j][0][0]
                depth_stop_h = blocks[i][j][0][1]
                depth_start_w = blocks[i][j][1][0]
                depth_stop_w = blocks[i][j][1][1]
                if depth_start_h <= i and i < depth_stop_h - factor_h:
                    if depth_start_w <= j and j < depth_stop_w - factor_w:
                        grid[i][j].append(0)
                        blocks[i][j] = (
                            (depth_start_h, depth_stop_h - factor_h),
                            (depth_start_w, depth_stop_w - factor_w)
                        )
                    else:
                        grid[i][j].append(1)
                        blocks[i][j] = (
                            (depth_start_h, depth_stop_h - factor_h),
                            (depth_start_w + factor_w, depth_stop_w)
                        )
                else:
                    if depth_start_w <= j and j < depth_stop_w - factor_w:
                        grid[i][j].append(2)
                        blocks[i][j] = (
                            (depth_start_h + factor_h, depth_stop_h),
                            (depth_start_w, depth_stop_w - factor_w)
                        )
                    else:
                        grid[i][j].append(3)
                        blocks[i][j] = (
                            (depth_start_h + factor_h, depth_stop_h),
                            (depth_start_w + factor_w, depth_stop_w)
                        )

    elements_added = 0
    for h in range(height):
        for w in range(width):
            for d in range(1, depth + 1):
                indices[h][w][d] = indices[h][w][d - 1] * 4 + grid[h][w][d]
            grid_list_of_lists_of_dicts[h][w]["grid_global_idx_stack"] = indices[h][w][:-1]
            grid_list_of_lists_of_dicts[h][w]["grid_local_idx_stack"] = grid[h][w][1:]
            if grid_list_of_lists_of_dicts[h][w].get("batch_idx") is None:
                grid_list_of_lists_of_dicts[h][w]["batch_idx"] = (
                    -(elements_added // batch_size + 1)
                )
                elements_added += 1

    grid_list_flat = [item for sublist in grid_list_of_lists_of_dicts for item in sublist]
    return grid_list_flat


def wsi_pre_inference(
    wsi_stitch_gcs_path,
    target_image_width,
    patch_height,
    patch_width,
    thumbnail_method,
    rgb2hed_threshold,
    include_patch_threshold,
    batch_size
):
    """Pre-inference setup for patch extraction and 4-ary tree traversal.

    Args:
        wsi_stitch_gcs_path: str, GCS path of WSI file.
        target_image_width: int, the approximate width of resultant thumbnail
            image.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        thumbnail_method: str, method to use for converting thumbnail of slide
            into binary mask. Either otsu or rgb2hed.
        rgb2hed_threshold: float, threshold to convert RGB2HED image to binary
            mask.
        include_patch_threshold: float, threshold to compare with percent of
            binary flags within a patch region to include in collection.
        batch_size: int, number of images to include in each batch for
            inference.

    Yields:
        Dictionary containing batch index, 4-ary tree indices and possibly
            coordinates and filename of patch PNG images to include in
            collection.
    """
    wsi, thumbnail = get_wsi_thumbnail(
        wsi_stitch_gcs_path, target_image_width
    )
    if thumbnail_method == "otsu":
        binary_img = otsu_method(thumbnail=thumbnail)
    else:
        binary_img = rgb2hed_method(
            thumbnail=thumbnail, threshold=rgb2hed_threshold
        )

    full_slide_mask = create_full_slide_mask(
        binary_img, slide_dims=wsi.level_dimensions[0]
    )
    max_dim = 2 ** max(
        math.ceil(math.log(wsi.level_dimensions[0][1] / patch_height, 2)),
        math.ceil(math.log(wsi.level_dimensions[0][0] / patch_width, 2))
    )

    grid_list_flat = wsi_build_grid(
        max_dim,
        wsi.level_dimensions[0],
        patch_height,
        patch_width,
        full_slide_mask,
        include_patch_threshold,
        batch_size
    )

    for grid_dict in grid_list_flat:
        coords = grid_dict.get("coords")
        if coords is None:
            yield (
                grid_dict["batch_idx"],
                {
                    "grid_global_idx_stack": grid_dict[
                        "grid_global_idx_stack"],
                    "grid_local_idx_stack": grid_dict["grid_local_idx_stack"],
                }
            )
        else:
            yield (
                grid_dict["batch_idx"],
                {
                    "grid_global_idx_stack": grid_dict[
                        "grid_global_idx_stack"],
                    "grid_local_idx_stack": grid_dict["grid_local_idx_stack"],
                    "coords": (coords[1], coords[0]),
                }
            )
