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

from collections import defaultdict
import cv2
from google.cloud import storage
import json
import math
import numpy as np
import os
import skimage.measure
import skimage.morphology
import tensorflow as tf


def build_output_type_set(output_type_list, config):
    """Builds set of output types.

    Args:
        output_type_list: list, possible output segmentation coord types.
        config: dict, user passed parameters.

    Returns:
        Set of requested output segmenation coord types.
    """
    output_types = set()
    for output_type in output_type_list:
        if config["output_{}".format(output_type)]:
            output_types.add(output_type)
    return output_types


def get_segmentation_coords_from_gcs(gcs_path):
    """Gets segmentation coordinates from GCS.

    Args:
        gcs_path: str, GCS path coordinate jsonl is stored.

    Returns:
        Dictionary with keys being coordinate tuples and values are lists of
            polygon vertex coordinates.
    """
    # Instantiate a Google Cloud Storage client.
    storage_client = storage.Client()
    # Specify required bucket and file.
    bucket = storage_client.get_bucket(gcs_path.split("/")[2])
    blob = bucket.blob("/".join(gcs_path.split("/")[3:]))

    # Download the contents of the blob as a string.
    data = blob.download_as_string(client=None)
    data_string = data.decode("utf-8")
    data_lines = data_string.split("\n")[:-1]
    data_dict = {}
    # Parse each line using json.loads() method.
    for line in data_lines:
        data_dict.update(json.loads(line))
    return data_dict


def get_patch_coords(filename):
    """Gets patch coordinates from patch filename.

    Args:
        filename: str, GCS filename of patch image.

    Returns:
        2-tuple of x and y coordinates of patch wrt. the slide's dimensions.
    """
    string_split = filename.split("/")[-1].split("_")
    x = float(string_split[2])
    y = float(string_split[4])
    return (x, y)


def build_patch_coord_list(file_list):
    """Builds patch coordinate list from patch filenames.

    Args:
        file_list: list, GCS filenames of patch images.

    Returns:
        List of patch coordinate 2-tuples.
    """
    return [get_patch_coords(filename) for filename in file_list]


class RestoredModel(object):
    """Class restores trained model and performs inference through sess.

    Attributes:
        graph: `tf.Graph`, holds the TensorFlow execution graph.
        sess: `tf.Session`, session to execute TensorFlow graph within.
        model_saver: `tf.compat.v1.train.Saver`, restores trained model into
            session.
        sample_in: tensor, input tensor of segmentation graph.
        c_mask_out: tensor, output tensor of segmentation graph.
    """
    def __init__(self, model_name, model_folder):
        """Constructor of `RestoredModel`.

        Args:
            model_name: str, the full name of the segmentation model.
            model_folder: str, the filepath of the segmentation model.
        """
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)

        with self.graph.as_default():
            self.model_saver = tf.compat.v1.train.import_meta_graph(
                meta_graph_or_file=model_name
            )
            self.model_saver.restore(
                sess=self.sess,
                save_path=tf.compat.v1.train.latest_checkpoint(
                    checkpoint_dir=model_folder
                )
            )
            self.graph = self.graph
            self.sample_in = self.graph.get_tensor_by_name(name="sample:0")
            self.c_mask_out = self.graph.get_tensor_by_name(name="c_mask:0")

    def run_sess(self, patches):
        """Runs patches through restored trained model session.

        Args:
            patches: np.array of image patches of shape
                (min(remaining, group_size), patch_size, patch_size, num_channels).

        Returns:
            np.array of segmented iamge of shape
                (min(remaining, group_size), patch_size, patch_size, 1).
        """
        feed_dict = {self.sample_in: patches}
        generated_mask = self.sess.run(
            fetches=[self.c_mask_out], feed_dict=feed_dict
        )
        return generated_mask

    def close_sess(self):
        """Closes TensorFlow session."""
        self.sess.close()


def image2patch(
    in_image,
    patch_size,
    stride,
    median_blur_image=False,
    median_blur_kernel_size=9
):
    """Converts input image to a list of patches.

    Args:
        in_image: tensor, image tensor of shape (height, width, 3).
        patch_size: int, the size of each square patch.
        stride: int, the number of pixels to jump for the next patch.
        median_blur_image: bool, whether to apply `medianBlur` to the input
            image.
        median_blur_kernel_size: int, the kernel size for `medianBlur`.

    Returns:
        np.array of image patches of shape
            (patch_size, patch_size, num_channels).
    """
    if median_blur_image is True:
        in_image = cv2.medianBlur(in_image, median_blur_kernel_size)
    shape = in_image.shape
    if shape[0] < patch_size:
        H = 0
    else:
        H = math.ceil((shape[0] - patch_size) / stride)
    if shape[1] < patch_size:
        W = 0
    else:
        W = math.ceil((shape[1] - patch_size) / stride)
    patch_list = []

    hpad = patch_size + stride * H - shape[0]
    wpad = patch_size + stride * W - shape[1]
    if len(shape) > 2:
        full_image = np.pad(
            in_image, ((0, hpad), (0, wpad), (0, 0)), mode='symmetric'
        )
    else:
        full_image = np.pad(
            in_image, ((0, hpad), (0, wpad)), mode='symmetric'
        )
    for i in range(H + 1):
        hs = i * stride
        he = i * stride + patch_size
        for j in range(W + 1):
            ws = j * stride
            we = j * stride + patch_size
            if len(shape) > 2:
                # element.shape = (patch_size, patch_size, 3)
                patch_list.append(full_image[hs:he, ws:we, :])
            else:
                # element.shape = (patch_size, patch_size)
                patch_list.append(full_image[hs:he, ws:we])
    if len(patch_list) != (H + 1) * (W + 1):
        raise ValueError(
            'Patch_list: ', str(len(patch_list), ' H: ', str(H), ' W: ', str(W))
        )
    # len = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1)
    return patch_list


def list2batch(patches):
    """Converts list of patches to a batch of patches.

    Args:
        patches: list, image patches of shape
            (patch_height, patch_width, 3).

    Returns:
        np.array of batch of image patches of shape
            (min(remaining, group_size), patch_size, patch_size, num_channels).
    """
    # (patch_size, patch_size, num_channels).
    patch_shape = list(patches[0].shape)
    # min(remaining, group_size).
    batch_size = len(patches)
    if len(patch_shape) > 2:
        batch = np.zeros([batch_size] + patch_shape)
        for index, temp in enumerate(patches):
            # shape = (min(remaining, group_size), patch_size, patch_size, num_channels).
            batch[index, :, :, :] = temp
    else:
        batch = np.zeros([batch_size] + patch_shape + [1])
        for index, temp in enumerate(patches):
            # shape = (min(remaining, group_size), patch_size, patch_size, 1).
            batch[index, :, :, :] = np.expand_dims(temp, axis=-1)
    return batch


def preprocess(input_image, config):
    """Preprocesses input image and batches images.

    Args:
        input_image: tensor, image tensor of shape
            (patch_height, patch_width, 3).
        config: dict, user passed parameters.

    Returns:
        List of length num_group of patch image arrays of shape
            (min(remaining, group_size), patch_size, patch_size, num_channels).
    """
    # len = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1)
    # Each element has shape = (patch_size, patch_size, num_channels).
    patch_list = image2patch(
        in_image=tf.cast(x=input_image, dtype=tf.float32) / 255.0,
        patch_size=config["segmentation_patch_size"],
        stride=config["segmentation_stride"],
        median_blur_image=config["segmentation_median_blur_image"],
        median_blur_kernel_size=config["segmentation_median_blur_kernel_size"]
    )
    # (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1) / group_size
    num_group = math.ceil(len(patch_list) / config["segmentation_group_size"])
    batch_group = []
    for i in range(num_group):
        start_idx = i * config["segmentation_group_size"]
        end_idx = (i + 1) * config["segmentation_group_size"]
        # shape = (min(remaining, group_size), patch_size, patch_size, num_channels).
        temp_batch = list2batch(patch_list[start_idx: end_idx])
        batch_group.append(temp_batch)
    return batch_group


def batch2list(batch):
    """Converts a batch of patches into a list of batches.

    Args:
        restored_model: Restored TensorFlow segmentation model.
        batch: np.array of shape
            (min(remaining, group_size), patch_size, patch_size, num_channels).

    Returns:
        List of length batch.shape[0] of patch image arrays of shape
            (patch_size, patch_size).
    """
    return [batch[index, :, :] for index in range(batch.shape[0])]


def sess_inference(restored_model, batch_group):
    """Inferences model session for each patch in batch group.

    Args:
        restored_model: Restored TensorFlow segmentation model.
        batch_group: list, length of num_group, contains batches of patches of
            shape
            (min(remaining, group_size), patch_size, patch_size, num_channels).

    Returns:
        List of segmented patches.
    """
    patch_list = []
    # len(batch_group) = num_group = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1) / group_size
    # temp_batch.shape = (min(remaining, group_size), patch_size, patch_size, num_channels).
    for temp_batch in batch_group:
        # shape = (min(remaining, group_size), patch_size, patch_size, 1).
        segmented_mask_batch = restored_model.run_sess(temp_batch)[0]
        # shape = (min(remaining, group_size), patch_size, patch_size).
        segmented_mask_batch = np.squeeze(segmented_mask_batch, axis=-1)
        # len(segmented_mask_list) = min(remaining, group_size)
        segmented_mask_list = batch2list(segmented_mask_batch)
        patch_list += segmented_mask_list
    # len(patch_list) = num_group = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1) / group_size
    return patch_list


def patch2image(patch_list, patch_size, stride, shape):
    """Combines patches from image back into full image.

    Args:
        patch_list: list, patch np.arrays of shape (patch_size, patch_size).
        patch_size: int, the size of each square patch.
        stride: int, the number of pixels to jump for the next patch.
        shape: tuple, the shape of the original image.

    Returns:
        np.array of combined patches into a single image.
    """
    if shape[0] < patch_size:
        H = 0
    else:
        H = math.ceil((shape[0] - patch_size) / stride)
    if shape[1] < patch_size:
        W = 0
    else:
        W = math.ceil((shape[1] - patch_size) / stride)

    # shape = (height, width).
    full_image = np.zeros([H * stride + patch_size, W * stride + patch_size])
    # shape = (height, width).
    bk = np.zeros([H * stride + patch_size, W * stride + patch_size])
    cnt = 0
    for i in range(H + 1):
        hs = i * stride
        he = hs + patch_size
        for j in range(W + 1):
            ws = j * stride
            we = ws + patch_size
            full_image[hs:he, ws:we] += patch_list[cnt]
            bk[hs:he, ws:we] += np.ones([patch_size, patch_size])
            cnt += 1
    full_image /= bk
    # numpy array shape = (height, width).
    image = full_image[0:shape[0], 0:shape[1]]
    return image


def center_point(mask):
    """Draws center point of segmentation mask.

    Args:
        mask: tensor, segmentation mask tensor of shape
            (patch_height, patch_width, 1).

    Returns:
        np.array center point of cell segmentation mask.
    """
    v, h = mask.shape
    center_mask = np.zeros([v, h])
    mask = skimage.morphology.erosion(mask, skimage.morphology.square(3))
    individual_mask = skimage.measure.label(mask, connectivity=2)
    prop = skimage.measure.regionprops(individual_mask)
    for cordinates in prop:
        temp_center = cordinates.centroid
        if not math.isnan(temp_center[0]) and not math.isnan(temp_center[1]):
            temp_mask = np.zeros([v, h])
            temp_mask[int(temp_center[0]), int(temp_center[1])] = 1
            center_mask += skimage.morphology.dilation(
                temp_mask, skimage.morphology.square(2)
            )
    return np.clip(center_mask, a_min=0, a_max=1).astype(np.uint8)


def draw_individual_edge(mask):
    """Draws individual edge from segmentation mask.

    Args:
        mask: tensor, segmentation mask tensor of shape
            (patch_height, patch_width, 1).

    Returns:
        np.array edge of cell segmentation mask.
    """
    v, h = mask.shape
    edge = np.zeros([v, h])
    individual_mask = skimage.measure.label(mask, connectivity=2)
    for index in np.unique(individual_mask):
        if index == 0:
            continue
        temp_mask = np.copy(individual_mask)
        temp_mask[temp_mask != index] = 0
        temp_mask[temp_mask == index] = 1
        temp_mask = skimage.morphology.dilation(
            temp_mask, skimage.morphology.square(3)
        )
        temp_edge = cv2.Canny(temp_mask.astype(np.uint8), 2, 5) / 255
        edge += temp_edge
    return np.clip(edge, a_min=0, a_max=1).astype(np.uint8)


def center_edge(mask, image):
    """Calculates centers and edges of cells from segmentation mask.

    Args:
        mask: tensor, segmentation mask tensor of shape
            (patch_height, patch_width, 1).
        image: tensor, image tensor of shape
            (patch_height, patch_width, 3).

    Returns:
        center_edge_masks: np.array, center and edge masks overlaid on
            original image.
        grayscale_maps: np.array, grayscale maps of centers and edges of
            cells.
    """
    # shape = (height, width).
    center_map = center_point(mask)
    # shape = (height, width).
    edge_map = draw_individual_edge(mask)
    # shape = (height, width).
    comb_mask = center_map + edge_map
    # shape = (height, width).
    comb_mask = np.clip(comb_mask, a_min=0, a_max=1)
    check_image = np.copy(image)
    comb_mask *= 255
    # shape = (height, width, num_channels).
    check_image[:, :, 1] = np.maximum(check_image[:, :, 1], comb_mask)
    return check_image.astype(np.uint8), comb_mask.astype(np.uint8)


def cell_seg_main(query_images, config):
    """Performs cell segmentation.

    Args:
        query_images: tensor, query image tensor of shape
            (batch, patch_height, patch_width, 3).
        config: dict, user passed parameters.

    Returns:
        center_edge_masks: list, center and edge masks overlaid on original
            image.
        grayscale_maps: list, grayscale maps of centers and edges of cells.
    """
    restored_model = RestoredModel(
        model_name=os.path.join(config["segmentation_export_dir"], config["segmentation_model_name"]),
        model_folder=config["segmentation_export_dir"]
    )
    center_edge_masks = []
    grayscale_maps = []
    for i, query_image in enumerate(query_images):
        batch_group = preprocess(query_image, config)
        mask_list = sess_inference(restored_model, batch_group)
        c_mask = patch2image(
            patch_list=mask_list,
            patch_size=config["segmentation_patch_size"],
            stride=config["segmentation_stride"],
            shape=query_image.shape
        )
        c_mask = cv2.medianBlur((255 * c_mask).astype(np.uint8), 3)
        c_mask = c_mask.astype(np.float) / 255
        thr = 0.5
        c_mask[c_mask < thr] = 0
        c_mask[c_mask >= thr] = 1
        center_edge_mask, grayscale_map = center_edge(c_mask, query_image)
        center_edge_masks.append(center_edge_mask)
        grayscale_maps.append(grayscale_map)

    restored_model.close_sess()
    # Lengths = query_image.shape[0], number of query images.
    return center_edge_masks, grayscale_maps


def mask_to_polygons(mask, min_area=10.):
    """Converts mask to polygons.

    Args:
        mask: tensor, query image tensor of shape
            (batch, patch_height, patch_width, 3).
        min_area: float, minimum amount of area within contour to be added as
            a polygon.

    Returns:
        `MultiPolygon` representation of input mask.
    """
    import shapely.geometry
    # First, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return shapely.geometry.MultiPolygon()
    # Now messy stuff to associate parent and child contours.
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # Create actual polygons filtering by area (removes artifacts).
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = shapely.geometry.Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = shapely.geometry.MultiPolygon(all_polygons)

    return all_polygons


def get_cell_polygon_coords(query_images, config):
    """Gets cell polygon coordinates.

    Args:
        query_images: tensor, query image tensor of shape
            (batch, patch_height, patch_width, 3).
        config: dict, user passed parameters.

    Returns:
        cell_coords_list: list, dictionaries containing lists of cell
            polygon exterior coordinates.
        nuclei_coords_list: list, dictionaries containing lists of cell
            polygon centroid coordinates.
        """
    _, grayscale_maps = cell_seg_main(query_images, config)
    cell_coords_list = []
    nuclei_coords_list = []
    for grayscale_map in grayscale_maps:
        cell_polygons = mask_to_polygons(grayscale_map)

        cell_coords_list.append(
            [list(polygon.exterior.coords) for polygon in cell_polygons]
        )
        nuclei_coords_list.append(
            [
                list(polygon.centroid.coords)[0]
                for polygon in cell_polygons
            ]
        )

    return cell_coords_list, nuclei_coords_list


def inference_from_saved_model(patch_coords_list, query_images, config):
    """Inferences segmentation SavedModel and gets outputs.

    Args:
        patch_coords_list: list, 2-tuples of the x and y-coordinates of each
            patch.
        query_images: tensor, query image tensor of shape
            (batch, patch_height, patch_width, 3).
        config: dict, user passed parameters.

    Returns:
        Dictionary of cell polygon exterior and centroid coordinate lists.
    """
    cell_coords_list, nuclei_coords_list = get_cell_polygon_coords(
        query_images, config
    )
    cell_coords_dict_list = []
    nuclei_coords_dict_list = []
    for i, coords in enumerate(patch_coords_list):
        cell_coords_dict_list.append(
            {str(coords): cell_coords_list[i]}
        )
        nuclei_coords_dict_list.append(
            {str(coords): nuclei_coords_list[i]}
        )

    return {
        "segmentation_cell_coords": cell_coords_dict_list,
        "segmentation_nuclei_coords": nuclei_coords_dict_list
    }
