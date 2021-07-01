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

import apache_beam as beam
import argparse
import datetime
import logging
import os

from components import confusion_matrix
from components import polygon
from components import pre_polygon


def run(argv=None, save_main_session=True):
    """Runs the polygon confusion matrix pipeline."""
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def str_to_float_list(s):
        return [
            float(x) for x in s.split(",")
        ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        type=str_to_float_list,
        required=True,
        help="List of float thresholds.")
    parser.add_argument(
        "--dilation_factors",
        type=str_to_float_list,
        required=True,
        help="List of float dilation factors.")
    parser.add_argument(
        "--slide_name",
        type=str,
        required=True,
        help="The name of the slide.")
    parser.add_argument(
        "--annotations_image_gcs_path",
        type=str,
        required=True,
        help="The GCS path to read annotations image.")
    parser.add_argument(
        "--kde_gs_image_gcs_path",
        type=str,
        required=True,
        help="The GCS path to read KDE GS image.")
    parser.add_argument(
        "--patch_coordinates_gcs_path",
        type=str,
        required=True,
        help="The GCS path to read patch coordinates.")
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
        "--image_height",
        type=int,
        default=8192,
        help="Number of pixels for full image's height.")
    parser.add_argument(
        "--image_width",
        type=int,
        default=8192,
        help="Number of pixels for full image's width.")
    parser.add_argument(
        "--effective_slide_height",
        type=int,
        default=2 ** 7 * 1024,
        help="Number of pixels for effective full slide's height.")
    parser.add_argument(
        "--effective_slide_width",
        type=int,
        default=2 ** 7 * 1024,
        help="Number of pixels for effective full slide's width.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=100,
        help="Number of seconds to timeout polygon math.")
    parser.add_argument(
        "--output_gcs_path",
        type=str,
        required=True,
        help="The GCS path to write confusion matrices.")
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

    job_name = "polygon-" + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

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

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(known_args.runner, options=pipeline_options) as p:
        pre_poly = (
            p |
            "{} Pre-polygons".format(known_args.slide_name) >> beam.Create(
                pre_polygon.pre_polygons(
                    thresholds=known_args.thresholds,
                    dilation_factors=known_args.dilation_factors
                )
             )
        )

        polygon_do = pre_poly | "Polygons" >> beam.ParDo(
            polygon.PolygonDoFn(
                slide_name=known_args.slide_name,
                annotations_image_gcs_path=(
                    known_args.annotations_image_gcs_path
                ),
                kde_gs_image_gcs_path=known_args.kde_gs_image_gcs_path,
                patch_coordinates_gcs_path=(
                    known_args.patch_coordinates_gcs_path
                ),
                patch_height=known_args.patch_height,
                patch_width=known_args.patch_width,
                height_scale_factor=(
                    known_args.image_height / known_args.effective_slide_height
                ),
                width_scale_factor=(
                    known_args.image_width / known_args.effective_slide_width
                ),
                timeout=known_args.timeout
            )
        )

        # Confusion matrix.
        confusion_matrix_to_csv_map = (
            polygon_do | "Confusion matrix to CSV" >> beam.FlatMap(
                confusion_matrix.confusion_matrix_to_csv
            )
        )
        write_confusion_matrix = (
            confusion_matrix_to_csv_map |
            "Write confusion matrix" >> beam.io.Write(
                beam.io.WriteToText(
                    file_path_prefix="{}_polygon_confusion_matrix.csv".format(
                        known_args.output_gcs_path
                    ),
                    num_shards=1
                )
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
