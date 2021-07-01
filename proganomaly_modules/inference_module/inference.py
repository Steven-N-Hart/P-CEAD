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

from proganomaly_modules.inference_module import beam_stitch
from proganomaly_modules.inference_module import image_utils
from proganomaly_modules.inference_module import gan_inference
from proganomaly_modules.inference_module import polygon
from proganomaly_modules.inference_module import segmentation_inference


def get_precomputed_outputs_from_gcs(config):
    """Gets precomputed inference outputs from GCS.

    Args:
        config: dict, user passed parameters.

    Returns:
        Dictionary of outputs.
    """
    (gan_output_type_list,
     segmentation_output_type_list,
     gs_image_set) = get_output_collections()

    output_dict = {}
    print("Loading pre-computed outputs!")
    # Polygons.
    if config["output"].get("output_kde_gs_polygon"):
        required_for_polygon = [
            config["input"].get("pre_computed_query_images_gcs_path"),
            config["input"].get("pre_computed_kde_gs_gcs_path"),
            config["input"].get("pre_computed_patch_coordinates_gcs_path")
        ]
        assert all(required_for_polygon), "To output KDE grayscale polygon, need query_images, kde_gs, and patch_coordinates pre-computed GCS paths"

        if not config["output"].get("output_query_images"):
            print(
                "Setting output config's {} to True since we need {}".format(
                    "output_query_images", "kde_gs_polygon"
                )
            )
            config["output"]["output_query_images"] = True
        if not config["output"].get("output_kde_gs"):
            print(
                "Setting output config's {} to True since we need {}".format(
                    "output_kde_gs", "kde_gs_polygon"
                )
            )
            config["output"]["output_kde_gs"] = True
        if not config["output"].get("output_patch_coordinates"):
            print(
                "Setting output config's {} to True since we need {}".format(
                    "output_patch_coordinates", "kde_gs_polygon"
                )
            )
            config["output"]["output_patch_coordinates"] = True

    # GAN images.
    for gan_output_type in gan_output_type_list:
        if gan_output_type == "kde_gs_polygon":
            # Handled above.
            continue
        if config["output"]["output_{}".format(gan_output_type)]:
            path = "pre_computed_{}_gcs_path".format(gan_output_type)
            assert config["input"].get(path), "To output {}, need pre-computed GCS path".format(gan_output_type)
            image = tf.io.read_file(filename=config["input"][path])
            if gan_output_type in gs_image_set:
                channels = 1
            else:
                channels = 3
            output_dict[gan_output_type] = tf.stack(
                values=[
                    image_utils.scale_images(
                        images=tf.io.decode_png(
                            contents=image, channels=channels
                        )
                    )
                ],
                axis=0
            )

    # Segmentation outputs.
    for segmentation_type in segmentation_output_type_list:
        if config["output"]["output_{}".format(segmentation_type)]:
            path = "pre_computed_{}_gcs_path".format(segmentation_type)
            assert config["input"].get(path)
            output_dict[segmentation_type] = (
                segmentation_inference.get_segmentation_coords_from_gcs(
                    gcs_path=config["input"][path]
                )
            )

    if config["output"]["output_patch_coordinates"]:
        assert config["input"].get("pre_computed_patch_coordinates_gcs_path"), "To output patch coordinates, need pre-computed GCS path"
        output_dict["patch_coordinates"] = beam_stitch.get_patch_coords_from_gcs(
            gcs_path=config["input"]["pre_computed_patch_coordinates_gcs_path"]
        )
    print("Finished loading pre-computed outputs!")
    return output_dict


def get_output_collections():
    """Gets output collections

    Returns:
        gan_output_type_list: list, possible GAN output image types.
        segmentation_output_type_list: list, possible output segmentation coord
            types.
        gs_image_set: set, image output types that are grayscale, i.e. have
            only one channel.
    """
    gan_output_type_list = [
        "query_images",
        "generated_images",
        "encoded_generated_images",
        "query_encoded_images",
        "query_gen_encoded_images",
        "query_anomaly_images_linear_rgb",
        "query_anomaly_images_linear_gs",
        "query_mahalanobis_distance_images_linear",
        "query_pixel_anomaly_flag_images",
        "kde_rgb",
        "kde_gs",
        "kde_gs_thresholded",
        "kde_gs_polygon"
    ]

    segmentation_output_type_list = [
        "segmentation_cell_coords",
        "segmentation_nuclei_coords"
    ]

    gs_image_set = set(
        [
            "query_anomaly_images_linear_gs",
            "query_mahalanobis_distance_images_linear",
            "query_pixel_anomaly_flag_images",
            "kde_gs",
            "kde_gs_thresholded",
            "annotations"
        ]
    )

    return gan_output_type_list, segmentation_output_type_list, gs_image_set


def individual_pngs_inference(
    gan_output_types_set, segmentation_output_types_set, gs_image_set, config
):
    """Performs inference using individual PNG image files.

    Args:
        gan_output_types_set: set, GAN model output type requested as strings.
        segmentation_output_types_set: set, segmentation model output type
            requested as strings.
        gs_image_set: set, image output types that are grayscale, i.e. have
            only one channel.
        config: dict, user passed parameters.

    Returns:
        Dictionary of outputs.
    """
    output_dict = {}
    file_list = tf.io.gfile.glob(
        pattern=config["input"]["png_individual_gcs_glob_pattern"]
    )
    if any(
        [
            config["output"]["output_kde_gs_polygon"],
            config["output"]["output_patch_coordinates"],
            segmentation_output_types_set
        ]
    ):
        output_dict["patch_coordinates"] = (
            segmentation_inference.build_patch_coord_list(
                file_list
            )
        )
    print("Decoding individual PNG image files!")
    images = tf.stack(
        values=[
            image_utils.decode_png_image_file(
                filename=f,
                channels=config["inference"]["patch_depth"]
            )
            for f in file_list
        ],
        axis=0
    )
    images = tf.image.resize(
        images=images,
        size=(
            config["inference"]["patch_height"],
            config["inference"]["patch_width"]
        ),
        method="nearest"
    )
    print("Inferencing individual PNG images!")
    if gan_output_types_set:
        config["inference"]["gan"]["patch_depth"] = (
            config["inference"]["patch_depth"]
        )
        print("Calling GAN model!")
        output_dict.update(
            gan_inference.inference_from_saved_model(
                images,
                gan_output_types_set,
                gs_image_set,
                config["inference"]["gan"]
            )
        )
        for output in gan_output_types_set:
            image = output_dict.get(output)
            if image is not None:
                output_dict[output] = tf.image.flip_up_down(
                    image=output_dict[output]
                )
    if segmentation_output_types_set:
        print("Calling segmentation model!")
        output_dict.update(
            segmentation_inference.inference_from_saved_model(
                output_dict["patch_coordinates"],
                images,
                config["inference"]["segmentation"]
            )
        )
    print("Inference of individual PNG images complete!")
    return output_dict


def stitch_inference(
    gan_output_types_set, segmentation_output_types_set, gs_image_set, config
):
    """Performs inference using individual PNG image files.

    Args:
        gan_output_types_set: set, GAN model output type requested as strings.
        segmentation_output_types_set: set, segmentation model output type
            requested as strings.
        gs_image_set: set, image output types that are grayscale, i.e. have
            only one channel.
        config: dict, user passed parameters.

    Returns:
        Dictionary of outputs.
    """
    output_dict = {}
    print("Performing stitch Dataflow pipeline!")
    beam_stitch.call_beam_stitch_pipeline(config)
    print("Dataflow stitch pipeline complete!")
    output_dict.update(
        beam_stitch.get_beam_outputs(
            output_gcs_path=config["output"]["output_gcs_path"],
            gan_output_types_set=gan_output_types_set,
            segmentation_output_types_set=(
                segmentation_output_types_set
            ),
            gs_image_set=gs_image_set
        )
    )
    return output_dict


def get_non_precomputed_outputs(config):
    """Gets non-precomputed inference outputs.

    Args:
        config: dict, user passed parameters.

    Returns:
        Dictionary of outputs.
    """
    output_dict = {}
    use_individual_pngs = (
        True if config["input"]["png_individual_gcs_glob_pattern"]
        else False
    )
    stitch_pngs_from_gcs = (
        True if config["input"]["png_patch_stitch_gcs_glob_pattern"]
        else False
    )
    stitch_wsi_from_gcs = (
        True if config["input"]["wsi_stitch_gcs_path"] else False
    )

    stitch_from_gcs = stitch_pngs_from_gcs or stitch_wsi_from_gcs
    assert use_individual_pngs != stitch_from_gcs, (
        "Can't use individual PNGs AND stitch!"
    )
    if stitch_from_gcs:
        assert stitch_pngs_from_gcs != stitch_wsi_from_gcs, (
            "Can't use both types of stitching!"
        )

    (gan_output_type_list,
     segmentation_output_type_list,
     gs_image_set) = get_output_collections()

    gan_output_types_set = gan_inference.build_output_type_set(
        gan_output_type_list, config["output"]
    )

    segmentation_output_types_set = (
        segmentation_inference.build_output_type_set(
            segmentation_output_type_list, config["output"]
        )
    )

    if use_individual_pngs:
        inference_fn = individual_pngs_inference
    else:
        inference_fn = stitch_inference

    output_dict = inference_fn(
        gan_output_types_set,
        segmentation_output_types_set,
        gs_image_set,
        config
    )
    return output_dict


def get_inference_outputs(config):
    """Gets inference outputs.

    Args:
        config: dict, user passed parameters.

    Returns:
        Dictionary of outputs.
    """
    # Images in output_dict are all scaled [-1., 1.].
    if (config["input"]["use_pre_computed_gcs_paths"]):
        return get_precomputed_outputs_from_gcs(config)
    else:
        return get_non_precomputed_outputs(config)


def inference_pipeline(config):
    """Performs inference pipeline.

    Args:
        config: dict, user passed parameters.

    Returns:
        Dictionary of outputs.
    """
    # First get inference outputs from models.
    output_dict = get_inference_outputs(config)

    # Get query images from output dictionary.
    query_images = output_dict.get("query_images")

    # Get KDE grayscale images from output dictionary.
    kde_gs_images = output_dict.get("kde_gs")

    # Get patch coordinates from output dictionary.
    patch_coordinates = output_dict.get("patch_coordinates")

    # If we have the necessary ingredients, create polygons
    if (query_images is not None and
        kde_gs_images is not None and
        patch_coordinates is not None and
        config["output"]["output_kde_gs_polygon"]
       ):
        print("Creating polygons!")
        patch_polygons = None
        if config["polygon"][
            "limit_polygon_vertices_to_only_patches"]:
            height, width = output_dict["query_images"].shape[1:3]
            patch_polygons = polygon.create_patch_polygons(
                patch_coords_list=patch_coordinates,
                patch_height=config["inference"]["patch_height"],
                patch_width=config["inference"]["patch_width"],
                height_scale_factor=(
                    height / config["polygon"]["effective_slide_height"]
                ),
                width_scale_factor=(
                    width / config["polygon"]["effective_slide_width"]
                )
            )
        output_dict["kde_gs_polygon"] = [
            polygon.create_prediction_polygons(
                query_image=query_image,
                kde_gs_image=kde_gs_image,
                threshold=config["polygon"]["kde_gs_polygon_threshold"],
                dilation_factor=(
                    config["polygon"]["kde_gs_polygon_dilation_factor"]
                ),
                dilation_origin=config["polygon"]["dilation_origin"],
                patch_polygons=patch_polygons
            )
            for (query_image, kde_gs_image) in zip(
                query_images, kde_gs_images
            )
        ]
        print("Polygons created!")
    else:
        print("Skipping polygons since don't have all of the ingredients.")
    return output_dict
