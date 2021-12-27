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

from google.cloud import storage
import os
import subprocess

from proganomaly_modules.inference_module import image_utils
from proganomaly_modules.inference_module import segmentation_inference


def run_subprocess_commands(command_list):
    """Runs subprocess commands.

    Args:
        command_list: list, command strings to run.
    """
    try:
        p = subprocess.run(
            command_list, capture_output=True, check=True, text=True
        )
        if not isinstance(p, subprocess.CompletedProcess):
            p.wait()
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise


def call_beam_stitch_pipeline(config):
    """Calls beam stitch pipeline.

    Args:
        config: dict, user passed parameters.
    """
    # Create beam command using config.
    beam_command = """#!/bin/sh
python3 ../proganomaly_modules/beam_image_stitch/beam_image_stitch.py \\
--stitch_query_images={stitch_query_images} \\
--stitch_query_gen_encoded_images={stitch_query_gen_encoded_images} \\
--stitch_query_anomaly_images_linear_rgb={stitch_query_anomaly_images_linear_rgb} \\
--stitch_query_anomaly_images_linear_gs={stitch_query_anomaly_images_linear_gs} \\
--stitch_query_mahalanobis_distance_images_linear={stitch_query_mahalanobis_distance_images_linear} \\
--stitch_query_pixel_anomaly_flag_images={stitch_query_pixel_anomaly_flag_images} \\
--stitch_kde_rgb={stitch_kde_rgb} \\
--stitch_kde_gs={stitch_kde_gs} \\
--stitch_kde_gs_thresholded={stitch_kde_gs_thresholded} \\
--stitch_annotations=false \\
--output_segmentation_cell_coords={output_segmentation_cell_coords} \\
--output_segmentation_nuclei_coords={output_segmentation_nuclei_coords} \\
--output_patch_coordinates={output_patch_coordinates} \\
--slide_name={slide_name} \\
--png_patch_stitch_gcs_glob_pattern={png_patch_stitch_gcs_glob_pattern} \\
--wsi_stitch_gcs_path={wsi_stitch_gcs_path} \\
--target_image_width={target_image_width} \\
--patch_height={patch_height} \\
--patch_width={patch_width} \\
--patch_depth={patch_depth} \\
--thumbnail_method={thumbnail_method} \\
--rgb2hed_threshold={rgb2hed_threshold} \\
--include_patch_threshold={include_patch_threshold} \\
--batch_size={batch_size} \\
--gan_export_dir={gan_export_dir} \\
--gan_export_name={gan_export_name} \\
--generator_architecture={generator_architecture} \\
--berg_use_Z_inputs={berg_use_Z_inputs} \\
--berg_latent_size={berg_latent_size} \\
--berg_latent_mean={berg_latent_mean} \\
--berg_latent_stddev={berg_latent_stddev} \\
--bandwidth={bandwidth} \\
--kernel={kernel} \\
--metric={metric} \\
--xbins={xbins} \\
--ybins={ybins} \\
--min_neighborhood_count={min_neighborhood_count} \\
--connectivity={connectivity} \\
--min_anomaly_points_remaining={min_anomaly_points_remaining} \\
--scaling_power={scaling_power} \\
--scaling_factor={scaling_factor} \\
--cmap_str={cmap_str} \\
--dynamic_bandwidth_scale_factor={dynamic_bandwidth_scale_factor} \\
--max_anomaly_points_for_kde={max_anomaly_points_for_kde} \\
--kde_threshold={kde_threshold} \\
--annotation_patch_gcs_filepath="" \\
--num_confusion_matrix_thresholds=0 \\
--custom_mahalanobis_distance_threshold={custom_mahalanobis_distance_threshold} \\
--nary_tree_depth={nary_tree_depth} \\
--segmentation_export_dir={segmentation_export_dir} \\
--segmentation_model_name={segmentation_model_name} \\
--segmentation_patch_size={segmentation_patch_size} \\
--segmentation_stride={segmentation_stride} \\
--segmentation_median_blur_image={segmentation_median_blur_image} \\
--segmentation_median_blur_kernel_size={segmentation_median_blur_kernel_size} \\
--segmentation_group_size={segmentation_group_size} \\
--output_patch_coordinates={output_patch_coordinates} \\
--output_gcs_path={output_gcs_path} \\
--project={project} \\
--bucket={bucket} \\
--region={region} \\
--autoscaling_algorithm={autoscaling_algorithm} \\
--num_workers={num_workers} \\
--machine_type={machine_type} \\
--disk_size_gb={disk_size_gb} \\
--service_account_email={service_account_email} \\
--use_public_ips={use_public_ips} \\
--network={network} \\
--subnetwork={subnetwork} \\
--runner=DataflowRunner \\
--setup_file=./../proganomaly_modules/beam_image_stitch/setup.py
""".format(
        stitch_query_images=config["output"]["output_query_images"],
        stitch_query_gen_encoded_images=(
            config["output"]["output_query_gen_encoded_images"]
        ),
        stitch_query_anomaly_images_linear_rgb=(
            config["output"]["output_query_anomaly_images_linear_rgb"]
        ),
        stitch_query_anomaly_images_linear_gs=(
            config["output"]["output_query_anomaly_images_linear_gs"]
        ),
        stitch_query_mahalanobis_distance_images_linear=(
            config["output"][
                "output_query_mahalanobis_distance_images_linear"]
        ),
        stitch_query_pixel_anomaly_flag_images=(
            config["output"]["output_query_pixel_anomaly_flag_images"]
        ),
        stitch_kde_rgb=config["output"]["output_kde_rgb"],
        stitch_kde_gs=config["output"]["output_kde_gs"],
        stitch_kde_gs_thresholded=(
            config["output"]["output_kde_gs_thresholded"]
        ),
        output_segmentation_cell_coords=(
            config["output"]["output_segmentation_cell_coords"]
        ),
        output_segmentation_nuclei_coords=(
            config["output"]["output_segmentation_nuclei_coords"]
        ),
        output_patch_coordinates=config["output"]["output_patch_coordinates"],
        slide_name=config["input"]["slide_name"],
        png_patch_stitch_gcs_glob_pattern=(
            config["input"]["png_patch_stitch_gcs_glob_pattern"]
        ),
        wsi_stitch_gcs_path=config["input"]["wsi_stitch_gcs_path"],
        target_image_width=config["inference"]["gan"]["target_image_width"],
        patch_height=config["inference"]["patch_height"],
        patch_width=config["inference"]["patch_width"],
        patch_depth=config["inference"]["patch_depth"],
        thumbnail_method=config["inference"]["gan"]["thumbnail_method"],
        rgb2hed_threshold=config["inference"]["gan"]["rgb2hed_threshold"],
        include_patch_threshold=(
            config["inference"]["gan"]["include_patch_threshold"]
        ),
        batch_size=config["inference"]["batch_size"],
        gan_export_dir=config["inference"]["gan"]["gan_export_dir"],
        gan_export_name=config["inference"]["gan"]["gan_export_name"],
        generator_architecture=(
            config["inference"]["gan"]["generator_architecture"]
        ),
        berg_use_Z_inputs=(
            config["inference"]["gan"]["berg_use_Z_inputs"]
        ),
        berg_latent_size=(
            config["inference"]["gan"]["berg_latent_size"]
        ),
        berg_latent_mean=(
            config["inference"]["gan"]["berg_latent_mean"]
        ),
        berg_latent_stddev=(
            config["inference"]["gan"]["berg_latent_stddev"]
        ),
        bandwidth=config["inference"]["gan"]["bandwidth"],
        kernel=config["inference"]["gan"]["kernel"],
        metric=config["inference"]["gan"]["metric"],
        xbins=config["inference"]["gan"]["xbins"],
        ybins=config["inference"]["gan"]["ybins"],
        min_neighborhood_count=(
            config["inference"]["gan"]["min_neighborhood_count"]
        ),
        connectivity=config["inference"]["gan"]["connectivity"],
        min_anomaly_points_remaining=(
            config["inference"]["gan"]["min_anomaly_points_remaining"]
        ),
        scaling_power=config["inference"]["gan"]["scaling_power"],
        scaling_factor=config["inference"]["gan"]["scaling_factor"],
        cmap_str=config["inference"]["gan"]["cmap_str"],
        dynamic_bandwidth_scale_factor=config["inference"]["gan"][
            "dynamic_bandwidth_scale_factor"],
        max_anomaly_points_for_kde=config["inference"]["gan"][
            "max_anomaly_points_for_kde"],
        kde_threshold=config["inference"]["gan"]["kde_threshold"],
        custom_mahalanobis_distance_threshold=(
            config["inference"]["gan"]["custom_mahalanobis_distance_threshold"]
        ),
        nary_tree_depth=config["inference"]["gan"]["nary_tree_depth"],
        segmentation_export_dir=config["inference"]["segmentation"][
            "segmentation_export_dir"],
        segmentation_model_name=config["inference"]["segmentation"][
            "segmentation_model_name"],
        segmentation_patch_size=config["inference"]["segmentation"][
            "segmentation_patch_size"],
        segmentation_stride=config["inference"]["segmentation"][
            "segmentation_stride"],
        segmentation_median_blur_image=config["inference"]["segmentation"][
            "segmentation_median_blur_image"],
        segmentation_median_blur_kernel_size=config["inference"]["segmentation"][
            "segmentation_median_blur_kernel_size"],
        segmentation_group_size=config["inference"]["segmentation"][
            "segmentation_group_size"],
        output_gcs_path=config["output"]["output_gcs_path"],
        project=config["dataflow"]["project"],
        bucket=config["dataflow"]["bucket"],
        region=config["dataflow"]["region"],
        autoscaling_algorithm=config["dataflow"]["autoscaling_algorithm"],
        num_workers=config["dataflow"]["num_workers"],
        machine_type=config["dataflow"]["machine_type"],
        disk_size_gb=config["dataflow"]["disk_size_gb"],
        service_account_email=config["dataflow"]["service_account_email"],
        use_public_ips=config["dataflow"]["use_public_ips"],
        network=config["dataflow"]["network"],
        subnetwork=config["dataflow"]["subnetwork"]
    )

    # Write beam command to shell file.
    shell_file = "run_beam.sh"
    with open(shell_file, "w") as f:
        f.write(beam_command)

    assert os.path.isfile(shell_file) and os.access(shell_file, os.R_OK), (
        "File is missing or is not readble.")

    # Call shell file in subprocess and wait for completion.
    run_subprocess_commands(command_list=["chmod", "777", shell_file])
    run_subprocess_commands(command_list=["./{}".format(shell_file)])


def get_patch_coords_from_gcs(gcs_path):
    """Gets patch coordinates from GCS.

    Args:
        gcs_path: str, GCS path patch coordinate CSV is stored.

    Returns:
        List of 2-tuples of x and y patch corner coordinates.
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
    data_list = []
    # Parse each line into a 2-tuple of int coordinates.
    return [
        (int(line.split(",")[0]), int(line.split(",")[1]))
        for line in data_lines
    ]


def get_beam_outputs(
    output_gcs_path,
    gan_output_types_set,
    segmentation_output_types_set,
    gs_image_set
):
    """Calls beam stitch pipeline.

    Args:
        output_gcs_path: str, GCS path to Dataflow pipeline's output images.
        gan_output_types_set: set, GAN model output types requested.
        segmentation_output_types_set: set, segmentation model output types
            requested.
        gs_image_set: set, image output types that are grayscale, i.e. have
            only one channel.

    Returns:
        Dictionary of outputs.
    """
    output_dict = {}
    # GAN.
    for output_type in gan_output_types_set:
        output_dict[output_type] = image_utils.scale_images(
            images=image_utils.decode_png_image_file(
                filename="{}_{}".format(output_gcs_path, output_type),
                channels=1 if output_type in gs_image_set else 3
            )
        )

    # Segmentation.
    for output_type in segmentation_output_types_set:
        output_dict[output_type] = (
            segmentation_inference.get_segmentation_coords_from_gcs(
                gcs_path="{}_{}".format(output_gcs_path, output_type)
            )
        )

    # Patch coordinates.
    output_dict["patch_coordinates"] = get_patch_coords_from_gcs(
        gcs_path="{}_{}".format(
            output_gcs_path, "patch_coordinates.csv-00000-of-00001"
        )
    )
    return output_dict
