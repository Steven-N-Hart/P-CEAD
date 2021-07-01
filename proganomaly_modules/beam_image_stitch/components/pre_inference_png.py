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

import math
import numpy as np
import tensorflow as tf


def get_file_params(png_patch_stitch_gcs_glob_pattern, patch_height, patch_width):
    """Gets PNG patch file parameters.

    Args:
        png_patch_stitch_gcs_glob_pattern: str, GCS path of patch PNG files.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.

    Returns:
        full_images_dim: int, the max number of patches between both
            dimensions.
        coords_list: list, tuples of (x, y, f) where x is the x-dimension
            coordinate, y is the y-dimension coordinate, and f is the filename
            of that corresponding patch PNG file.
    """
    filename_list = tf.io.gfile.glob(png_patch_stitch_gcs_glob_pattern)
    coords_list = []
    if filename_list:
        for i, split in enumerate(filename_list[0].split("_")):
            if split == "x":
                x_coord_idx = i
                break
    for file in filename_list:
        file_split = file.split("_")
        x, y, width, height = (
            file_split[x_coord_idx + 1],
            file_split[x_coord_idx + 3],
            file_split[x_coord_idx + 5],
            file_split[x_coord_idx + 7].split(".")[0]
        )
        # Switch x & y coords due to weird way these are represented!
        coords_list.extend([(int(y), int(x), file)])

    coords_arr = np.array([(x, y) for x, y, _ in coords_list])
    x_min, x_max = np.min(coords_arr[:, 0]), np.max(coords_arr[:, 0])
    x_range = x_max - x_min
    y_min, y_max = np.min(coords_arr[:, 1]), np.max(coords_arr[:, 1])
    y_range = y_max - y_min
    num_images_x, num_images_y = (
        x_range // patch_height + 1, y_range // patch_width + 1
    )

    for i, coords in enumerate(coords_list):
        x = (coords[0] - x_min) // patch_height
        y = (coords[1] - y_min) // patch_width
        f = coords[2]
        coords_list[i] = (x, y, f)
    full_images_x = int(math.pow(2, math.ceil(math.log(num_images_x, 2))))
    full_images_y = int(math.pow(2, math.ceil(math.log(num_images_y, 2))))
    full_images_dim = max(full_images_x, full_images_y)

    return full_images_dim, coords_list


def png_build_grid(max_dim, coords_list, batch_size):
    """Builds grid for 4-ary tree stitching of patches.

    Args:
        max_dim: int, the maximum dimension between height and width.
        coords_list: list, tuples of (x, y, f) where x is the x-dimension
            coordinate, y is the y-dimension coordinate, and f is the filename
            of that corresponding patch PNG file.
        batch_size: int, number of images to include in each batch for
            inference.

    Returns:
        List of dictionaries containing 4-ary tree indices and possibly
            coordinates of patch regions to include in collection.
    """
    grid_list_of_lists_of_dicts = [
        [
            {"height": i, "width": j}
            for j in range(max_dim)
        ]
        for i in range(max_dim)
    ]

    for i, coords in enumerate(coords_list):
        h = coords[0]
        w = coords[1]
        filename = coords[2]
        grid_list_of_lists_of_dicts[h][w]["filename"] = filename
        grid_list_of_lists_of_dicts[h][w]["coords"] = (coords[0], coords[1])
        grid_list_of_lists_of_dicts[h][w]["batch_idx"] = i // batch_size
        
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
    indices = [
        [[0] * (depth + 1) for j in range(width)] for i in range(height)
    ]
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
            grid_list_of_lists_of_dicts[h][w]["grid_global_idx_stack"] = (
                indices[h][w][:-1]
            )
            grid_list_of_lists_of_dicts[h][w]["grid_local_idx_stack"] = (
                grid[h][w][1:]
            )
            if grid_list_of_lists_of_dicts[h][w].get("batch_idx") is None:
                grid_list_of_lists_of_dicts[h][w]["batch_idx"] = (
                    -(elements_added // batch_size + 1)
                )
                elements_added += 1

    return grid_list_of_lists_of_dicts


def png_patch_pre_inference(
    png_patch_stitch_gcs_glob_pattern, patch_height, patch_width, batch_size
):
    """Pre-inference setup for PNG patch files and 4-ary tree traversal.

    Args:
        png_patch_stitch_gcs_glob_pattern: str, GCS path of patch PNG files.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        batch_size: int, number of images to include in each batch for
            inference.

    Yields:
        Dictionary containing batch index, 4-ary tree indices and possibly
            coordinates and filename of patch PNG images to include in
            collection.
    """
    max_dim, coords_list = get_file_params(
        png_patch_stitch_gcs_glob_pattern, patch_height, patch_width
    )
    grid_list_of_lists_of_dicts = png_build_grid(
        max_dim, coords_list, batch_size
    )
    grid_list_flat = [
        item for sublist in grid_list_of_lists_of_dicts for item in sublist
    ]
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
                    "coords": (
                        coords[1] * patch_height, coords[0] * patch_width
                    ),
                    "filename": grid_dict["filename"]
                }
            )
