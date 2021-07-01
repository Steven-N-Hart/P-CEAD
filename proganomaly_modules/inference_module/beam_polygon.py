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


def call_beam_polygon_pipeline(config):
    """Calls beam polygon pipeline.

    Args:
        config: dict, user passed parameters.
    """
    # Create beam command using config.
    beam_command = """#!/bin/sh
python3 ../proganomaly_modules/beam_polygon_confusion_matrix/polygon_threshold_dilation_confusion_matrix.py \\
--thresholds={thresholds} \\
--dilation_factors={dilation_factors} \\
--slide_name={slide_name} \\
--annotations_image_gcs_path={annotations_image_gcs_path} \\
--kde_gs_image_gcs_path={kde_gs_image_gcs_path} \\
--patch_coordinates_gcs_path={patch_coordinates_gcs_path} \\
--patch_height={patch_height} \\
--patch_width={patch_width} \\
--image_height={image_height} \\
--image_width={image_width} \\
--effective_slide_height={effective_slide_height} \\
--effective_slide_width={effective_slide_width} \\
--timeout={timeout} \\
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
--setup_file=./../proganomaly_modules/beam_polygon_confusion_matrix/setup.py
""".format(
        thresholds=",".join(
            [str(x) for x in config["polygon"]["thresholds"]]
        ),
        dilation_factors=",".join(
            [str(x) for x in config["polygon"]["dilation_factors"]]
        ),
        slide_name=config["input"]["slide_name"],
        annotations_image_gcs_path=(
            config["input"]["annotations_image_gcs_path"]
        ),
        kde_gs_image_gcs_path=config["input"]["kde_gs_image_gcs_path"],
        patch_coordinates_gcs_path=(
            config["input"]["patch_coordinates_gcs_path"]
        ),
        patch_height=config["polygon"]["patch_height"],
        patch_width=config["polygon"]["patch_width"],
        image_height=config["polygon"]["image_height"],
        image_width=config["polygon"]["image_width"],
        effective_slide_height=config["polygon"]["effective_slide_height"],
        effective_slide_width=config["polygon"]["effective_slide_width"],
        timeout=config["polygon"]["timeout"],
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
