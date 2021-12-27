# Copyright 2021 Google Inc. All Rights Reserved.
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

import apache_beam as beam
import argparse
import datetime
import logging
import os

from components import confusion_matrix
from components import images
from components import inference
from components import patch_coordinates
from components import pre_inference_png
from components import pre_inference_wsi
from components import segmentation


def run(argv=None, save_main_session=True):
    """Runs the patch inference stitching pipeline."""
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def _str_to_2_int_tuple_list(s):
        if not s: return []
        tuple_list = [
            tuple([int(z) for z in x.split(";")]) for x in s.split(",")
        ]
        for tup in tuple_list:
            assert len(tup) == 2, "Tuples should have two values: height & width."
        return tuple_list

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stitch_query_images",
        type=_str2bool,
        default=False,
        help="Whether to stitch query images.")
    parser.add_argument(
        "--stitch_query_gen_encoded_images",
        type=_str2bool,
        default=False,
        help="Whether to stitch query generator encoded images.")
    parser.add_argument(
        "--stitch_query_anomaly_images_linear_rgb",
        type=_str2bool,
        default=False,
        help="Whether to stitch query anomaly images linear RGB.")
    parser.add_argument(
        "--stitch_query_anomaly_images_linear_gs",
        type=_str2bool,
        default=False,
        help="Whether to stitch query anomaly images linear GS.")
    parser.add_argument(
        "--stitch_query_mahalanobis_distance_images_linear",
        type=_str2bool,
        default=False,
        help="Whether to stitch query Mahalanobis distance images linear.")
    parser.add_argument(
        "--stitch_query_pixel_anomaly_flag_images",
        type=_str2bool,
        default=False,
        help="Whether to stitch query pixel anomaly flag images.")
    parser.add_argument(
        "--stitch_kde_rgb",
        type=_str2bool,
        default=False,
        help="Whether to stitch KDE RGB images.")
    parser.add_argument(
        "--stitch_kde_gs",
        type=_str2bool,
        default=False,
        help="Whether to stitch KDE GS images.")
    parser.add_argument(
        "--stitch_kde_gs_thresholded",
        type=_str2bool,
        default=False,
        help="Whether to stitch KDE GS thresholded images.")
    parser.add_argument(
        "--stitch_annotations",
        type=_str2bool,
        default=False,
        help="Whether to stitch annotation images.")
    parser.add_argument(
        "--output_segmentation_cell_coords",
        type=_str2bool,
        default=False,
        help="Whether to output segmentation cell coordinates.")
    parser.add_argument(
        "--output_segmentation_nuclei_coords",
        type=_str2bool,
        default=False,
        help="Whether to output segmentation nuclei coordinates.")
    parser.add_argument(
        "--output_patch_coordinates",
        type=_str2bool,
        default=False,
        help="Whether to output patch coordinates.")
    parser.add_argument(
        "--slide_name",
        type=str,
        default="",
        help="Name of slide to stitch.")
    parser.add_argument(
        "--png_patch_stitch_gcs_glob_pattern",
        type=str,
        default="",
        help="GCS path of PNG patch images.")
    parser.add_argument(
        "--wsi_stitch_gcs_path",
        type=str,
        default="",
        help="GCS path of WSI.")
    parser.add_argument(
        "--target_image_width",
        type=int,
        default=500,
        help="The target thumbnail image width.")
    parser.add_argument(
        "--patch_height",
        type=int,
        default=1024,
        help="Number of pixels for patch's height.")
    parser.add_argument(
        "--patch_width",
        type=int,
        default=1024,
        help="Number of pixels for patch's width.")
    parser.add_argument(
        "--patch_depth",
        type=int,
        default=3,
        help="Number of channels for patch's depth.")
    parser.add_argument(
        "--thumbnail_method",
        type=str,
        default="otsu",
        help="Method to apply to thumbnail.")
    parser.add_argument(
        "--rgb2hed_threshold",
        type=float,
        default=-0.41,
        help="Threshold to use for RGB2HED thumbnail method.")
    parser.add_argument(
        "--include_patch_threshold",
        type=float,
        default=0.0,
        help="Threshold using thumbnail when to include a patch.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images to inference the model at once.")
    parser.add_argument(
        "--gan_export_dir",
        type=str,
        default="",
        help="Directory containing exported GAN models.")
    parser.add_argument(
        "--gan_export_name",
        type=str,
        default="",
        help="Name of directory of exported GAN model.")
    parser.add_argument(
        "--generator_architecture",
        type=str,
        default="GANomaly",
        help="Generator architecture type: 'berg' or 'GANomaly'.")
    parser.add_argument(
        "--berg_use_Z_inputs",
        type=_str2bool,
        default=False,
        help="For berg architecture, whether to use Z inputs. Query image inputs are always used.")
    parser.add_argument(
        "--berg_latent_size",
        type=int,
        default=512,
        help="For berg architecture, the latent size of the noise vector.")
    parser.add_argument(
        "--berg_latent_mean",
        type=float,
        default=0.0,
        help="For berg architecture, the latent vector's random normal mean.")
    parser.add_argument(
        "--berg_latent_stddev",
        type=float,
        default=1.0,
        help="For berg architecture, the latent vector's random normal standard deviation.")
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=100.,
        help="Bandwidth for kernel density estimation (KDE).")
    parser.add_argument(
        "--kernel",
        type=str,
        default="gaussian",
        help="Kernel method to use for kernel density estimation (KDE).")
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance method to use for kernel density estimation (KDE).")
    parser.add_argument(
        "--xbins",
        type=int,
        default=100,
        help="Number of sample bins in x-dimension for kernel density estimation (KDE).")
    parser.add_argument(
        "--ybins",
        type=int,
        default=100,
        help="Number of sample bins in y-dimension for kernel density estimation (KDE).")
    parser.add_argument(
        "--min_neighborhood_count",
        type=int,
        default=1,
        help="Minimum number of pixels within connected components' connectivity to not be removed before kernel density estimation (KDE).")
    parser.add_argument(
        "--connectivity",
        type=int,
        default=1,
        help="The adjacent connectivity between pixel flag connected components used for possible removal before kernel density estimation (KDE).")
    parser.add_argument(
        "--min_anomaly_points_remaining",
        type=int,
        default=1,
        help="The minimum number of anomaly points that must remain after small object removal to not zero out the kernel density estimation (KDE) image.")
    parser.add_argument(
        "--scaling_power",
        type=float,
        default=1.0,
        help="Exponent to transform KDE log densities.")
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=100000.0,
        help="Divisor factor to scale exponentiated KDE log densities.")
    parser.add_argument(
        "--cmap_str",
        type=str,
        default="turbo",
        help="Color map to use for RGB kernel density estimation (KDE) image.")
    parser.add_argument(
        "--dynamic_bandwidth_scale_factor",
        type=float,
        default=50000.0,
        help="Amount to scale the bandwidth based on anomaly counts.")
    parser.add_argument(
        "--max_anomaly_points_for_kde",
        type=int,
        default=50000,
        help="Maximum number of points allowed to run KDE. Otherwise entire image is marked as anomalous.")
    parser.add_argument(
        "--kde_threshold",
        type=float,
        default=0.2,
        help="Threshold for KDE GS image.")
    parser.add_argument(
        "--annotation_patch_gcs_filepath",
        type=str,
        default="",
        help="Input file pattern of images.")
    parser.add_argument(
        "--num_confusion_matrix_thresholds",
        type=int,
        default=0,
        help="Number of confusion matrix thresholds.")
    parser.add_argument(
        "--custom_mahalanobis_distance_threshold",
        type=float,
        default=-1.0,
        help="Custom Mahalanobis distance threshold.")
    parser.add_argument(
        "--nary_tree_depth",
        type=int,
        default=1,
        help="Depth of n-ary tree.")
    parser.add_argument(
        "--output_image_sizes",
        type=_str_to_2_int_tuple_list,
        default=[(1024, 1024)],
        help="List of 2-tuples of output image height and width for each n-ary level, starting from leaves.")
    parser.add_argument(
        "--segmentation_export_dir",
        type=str,
        default="",
        help="Directory containing exported segmentation models.")
    parser.add_argument(
        "--segmentation_model_name",
        type=str,
        default="",
        help="Name of segmentation model.")
    parser.add_argument(
        "--segmentation_patch_size",
        type=int,
        default=128,
        help="Size of each patch of image for segmentation model.")
    parser.add_argument(
        "--segmentation_stride",
        type=int,
        default=16,
        help="Number of pixels to skip for each patch of image for segmentation model.")
    parser.add_argument(
        "--segmentation_median_blur_image",
        type=_str2bool,
        default=False,
        help="Whether to median blur images before segmentation.")
    parser.add_argument(
        "--segmentation_median_blur_kernel_size",
        type=int,
        default=9,
        help="The kernel size of median blur for segmentation.")
    parser.add_argument(
        "--segmentation_group_size",
        type=int,
        default=10,
        help="Number of patches to include in a group for segmentation.")
    parser.add_argument(
        "--output_gcs_path",
        type=str,
        required=True,
        help="GCS file path to write outputs to.")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="The GCP project to use for the job.")
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="The GCS bucket to use for staging.")
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="The GCP region to use for the job.")
    parser.add_argument(
        "--autoscaling_algorithm",
        type=str,
        choices=["THROUGHPUT_BASED", "NONE"],
        default="THROUGHPUT_BASED",
        help="Input file pattern of images.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of Dataflow workers.")
    parser.add_argument(
        "--machine_type",
        type=str,
        default="n1-standard-1",
        help="The machine type of each Dataflow worker.")
    parser.add_argument(
        "--disk_size_gb",
        type=int,
        default=0,
        help="The disk size, in gigabytes, to use on each worker instance.")
    parser.add_argument(
        "--service_account_email",
        type=str,
        required=True,
        help="User-managed controller service account, using the format my-service-account-name@<project-id>.iam.gserviceaccount.com.")
    parser.add_argument(
        "--use_public_ips",
        type=_str2bool,
        default=False,
        help="Specifies that Dataflow workers must use public IP addresses. If the value is set to false, Dataflow workers use private IP addresses for all communication.")
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help="The Compute Engine network for launching Compute Engine instances to run your pipeline.")
    parser.add_argument(
        "--subnetwork",
        type=str,
        required=True,
        help="The Compute Engine subnetwork for launching Compute Engine instances to run your pipeline.")
    parser.add_argument(
        "--runner",
        default="DirectRunner",
        help="Type of runner.")
    known_args, pipeline_args = parser.parse_known_args(argv)
    logging.info("known_args = {}".format(known_args))
    logging.info("pipeline_args = {}".format(pipeline_args))

    use_png = True if known_args.png_patch_stitch_gcs_glob_pattern else False
    use_wsi = True if known_args.wsi_stitch_gcs_path else False

    assert use_png != use_wsi and (use_png or use_wsi)

    job_name = "stitch-" + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    # We use the save_main_session option because one or more DoFn"s in this
    # workflow rely on global context (e.g., a module imported at module level).
    options = {
        "job_name": job_name,
        "experiments": ["shuffle_mode=service"],
        "project": known_args.project,
        "staging_location": os.path.join(known_args.bucket, "tmp", "staging"),
        "temp_location": os.path.join(known_args.bucket, "tmp"),
        "region": known_args.region,
        "autoscaling_algorithm": known_args.autoscaling_algorithm,
        "num_workers": known_args.num_workers,
        "machine_type": known_args.machine_type,
        "disk_size_gb": known_args.disk_size_gb,
        "service_account_email": known_args.service_account_email,
        "use_public_ips": known_args.use_public_ips,
        "network": known_args.network,
        "subnetwork": known_args.subnetwork
    }
    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        flags=pipeline_args, **options
    )
    pipeline_options.view_as(
        beam.options.pipeline_options.SetupOptions
    ).save_main_session = save_main_session

    image_stitch_types_set = set()
    if known_args.stitch_query_images:
        image_stitch_types_set.add("query_images")
    if known_args.stitch_query_gen_encoded_images:
        image_stitch_types_set.add("query_gen_encoded_images")
    if known_args.stitch_query_anomaly_images_linear_rgb:
        image_stitch_types_set.add("query_anomaly_images_linear_rgb")
    if known_args.stitch_query_anomaly_images_linear_gs:
        image_stitch_types_set.add("query_anomaly_images_linear_gs")
    if known_args.stitch_query_mahalanobis_distance_images_linear:
        image_stitch_types_set.add("query_mahalanobis_distance_images_linear")
    if known_args.stitch_query_pixel_anomaly_flag_images:
        image_stitch_types_set.add("query_pixel_anomaly_flag_images")
    if known_args.stitch_kde_rgb:
        image_stitch_types_set.add("kde_rgb")
    if known_args.stitch_kde_gs:
        image_stitch_types_set.add("kde_gs")
    if known_args.stitch_kde_gs_thresholded:
        image_stitch_types_set.add("kde_gs_thresholded")
    if known_args.stitch_annotations:
        image_stitch_types_set.add("annotations")

    segmentation_coord_types_set = set()
    if known_args.output_segmentation_cell_coords:
        segmentation_coord_types_set.add("segmentation_cell_coords")
    if known_args.output_segmentation_nuclei_coords:
        segmentation_coord_types_set.add("segmentation_nuclei_coords")

    assert(len(known_args.output_image_sizes) >= known_args.nary_tree_depth)

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(known_args.runner, options=pipeline_options) as p:
        if use_wsi:
            pre_inf = (
                p |
                "{} WSI Pre-inference".format(
                    known_args.slide_name) >> beam.Create(
                    pre_inference_wsi.wsi_pre_inference(
                        wsi_stitch_gcs_path=known_args.wsi_stitch_gcs_path,
                        target_image_width=known_args.target_image_width,
                        patch_height=known_args.patch_height,
                        patch_width=known_args.patch_width,
                        thumbnail_method=known_args.thumbnail_method,
                        rgb2hed_threshold=known_args.rgb2hed_threshold,
                        include_patch_threshold=(
                            known_args.include_patch_threshold
                        ),
                        batch_size=known_args.batch_size
                    )
                 )
            )
        else:
            pre_inf = (
                p |
                "{} PNG Patch Pre-inference".format(
                    known_args.slide_name) >> beam.Create(
                    pre_inference_png.png_patch_pre_inference(
                        png_patch_stitch_gcs_glob_pattern=(
                            known_args.png_patch_stitch_gcs_glob_pattern
                        ),
                        patch_height=known_args.patch_height,
                        patch_width=known_args.patch_width,
                        batch_size=known_args.batch_size
                    )
                 )
            )

        if image_stitch_types_set or segmentation_coord_types_set:
            batch = pre_inf | "Group Batch Index" >> beam.GroupByKey()

            inference_do = batch | "Inference" >> beam.ParDo(
                inference.InferenceDoFn(
                    wsi_stitch_gcs_path=known_args.wsi_stitch_gcs_path,
                    patch_height=known_args.patch_height,
                    patch_width=known_args.patch_width,
                    patch_depth=known_args.patch_depth,
                    gan_export_dir=known_args.gan_export_dir,
                    gan_export_name=known_args.gan_export_name,
                    generator_architecture=known_args.generator_architecture,
                    berg_use_Z_inputs=known_args.berg_use_Z_inputs,
                    berg_latent_size=known_args.berg_latent_size,
                    berg_latent_mean=known_args.berg_latent_mean,
                    berg_latent_stddev=known_args.berg_latent_stddev,
                    image_stitch_types_set=image_stitch_types_set,
                    bandwidth=known_args.bandwidth,
                    kernel=known_args.kernel,
                    metric=known_args.metric,
                    xbins=known_args.xbins,
                    ybins=known_args.ybins,
                    min_neighborhood_count=known_args.min_neighborhood_count,
                    connectivity=known_args.connectivity,
                    min_anomaly_points_remaining=(
                        known_args.min_anomaly_points_remaining
                    ),
                    scaling_power=known_args.scaling_power,
                    scaling_factor=known_args.scaling_factor,
                    cmap_str=known_args.cmap_str,
                    dynamic_bandwidth_scale_factor=(
                        known_args.dynamic_bandwidth_scale_factor
                    ),
                    max_anomaly_points_for_kde=(
                        known_args.max_anomaly_points_for_kde
                    ),
                    kde_threshold=known_args.kde_threshold,
                    annotation_patch_gcs_filepath=(
                        known_args.annotation_patch_gcs_filepath
                    ),
                    num_confusion_matrix_thresholds=(
                        known_args.num_confusion_matrix_thresholds
                    ),
                    custom_mahalanobis_distance_threshold=(
                        known_args.custom_mahalanobis_distance_threshold
                    ),
                    segmentation_coord_types_set=segmentation_coord_types_set,
                    segmentation_export_dir=(
                        known_args.segmentation_export_dir
                    ),
                    segmentation_model_name=(
                        known_args.segmentation_model_name
                    ),
                    segmentation_patch_size=(
                        known_args.segmentation_patch_size
                    ),
                    segmentation_stride=known_args.segmentation_stride,
                    segmentation_median_blur_image=(
                        known_args.segmentation_median_blur_image
                    ),
                    segmentation_median_blur_kernel_size=(
                        known_args.segmentation_median_blur_kernel_size
                    ),
                    segmentation_group_size=known_args.segmentation_group_size
                )
            )

        # Images.
        for stitch_type in image_stitch_types_set:
            leaf_combine = (
                inference_do |
                "Leaf Combine_image_{}".format(stitch_type) >> beam.ParDo(
                    images.LeafCombineDoFn(stitch_type=stitch_type)
                )
            )
            leaf_group = (
                leaf_combine |
                "Leaf Group_{}".format(stitch_type) >> beam.GroupByKey()
            )
            branch_combine = (
                leaf_group |
                "Branch-leaf Combine_image_{}".format(
                    stitch_type
                ) >> beam.ParDo(
                    images.BranchCombineDoFn(
                        patch_height=known_args.output_image_sizes[0][0],
                        patch_width=known_args.output_image_sizes[0][1]
                    )
                )
            )
            for i in range(known_args.nary_tree_depth - 1):
                branch_group = (
                    branch_combine |
                    "Branch-branch Group image_{}_{}".format(
                        i, stitch_type) >> beam.GroupByKey()
                )
                branch_combine = (
                    branch_group |
                    "Branch-branch Combine image_{}_{}".format(
                        i, stitch_type) >> beam.ParDo(
                        images.BranchCombineDoFn(
                            patch_height=(
                                known_args.output_image_sizes[i + 1][0]
                            ),
                            patch_width=known_args.output_image_sizes[i + 1][1]
                        )
                    )
                )
            write_images = (
                branch_combine |
                "Write images_image_{}".format(stitch_type) >> beam.ParDo(
                    images.WriteImageDoFn(
                        output_filename="{}_{}".format(
                            known_args.output_gcs_path, stitch_type
                        )
                    )
                )
            )

        # Confusion matrix.
        if known_args.num_confusion_matrix_thresholds > 0:
            confusion_matrix_to_csv_map = (
                inference_do |
                "Confusion matrix to CSV" >> beam.FlatMap(
                    confusion_matrix.confusion_matrix_to_csv
                )
            )
            write_confusion_matrix = (
                confusion_matrix_to_csv_map |
                "Write confusion matrix CSV" >> beam.io.Write(
                    beam.io.WriteToText(
                        file_path_prefix="{}_{}.csv".format(
                            known_args.output_gcs_path, "confusion_matrix"
                        ),
                        num_shards=1
                    )
                )
            )

        # Segmentation coordinates.
        for segmentation_type in segmentation_coord_types_set:
            segmentation_dict_to_json_map = (
                inference_do |
                "Segmentation {} Dict to JSON".format(
                    segmentation_type) >> beam.FlatMap(
                    lambda x: segmentation.segmentation_dict_to_json(
                        element_dict=x, key=segmentation_type
                    )
                )
            )
            write_segmentation_coords = (
                segmentation_dict_to_json_map |
                "Write {} JSON".format(segmentation_type) >> beam.io.Write(
                    beam.io.WriteToText(
                        file_path_prefix="{}_{}.jsonl".format(
                            known_args.output_gcs_path, segmentation_type
                        ),
                        num_shards=1
                    )
                )
            )

        # Patch coordinates.
        if known_args.output_patch_coordinates:
            patch_coordinates_to_csv_map = (
                pre_inf |
                "Patch coordinates to CSV" >> beam.FlatMap(
                    patch_coordinates.patch_coordinates_to_csv
                )
            )
            write_patch_coordinates = (
                patch_coordinates_to_csv_map |
                "Write patch coordinates CSV" >> beam.io.Write(
                    beam.io.WriteToText(
                        file_path_prefix="{}_{}.csv".format(
                            known_args.output_gcs_path, "patch_coordinates"
                        ),
                        num_shards=1
                    )
                )
            )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
