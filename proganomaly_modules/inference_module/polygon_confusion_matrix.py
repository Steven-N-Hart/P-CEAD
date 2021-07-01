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

import os
import sys
import tensorflow as tf
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from proganomaly_modules.beam_polygon_confusion_matrix.components import confusion_matrix
from proganomaly_modules.beam_polygon_confusion_matrix.components import polygon
from proganomaly_modules.inference_module import beam_polygon


def save_polygon_confusion_matrices(config):
    """Saves polygon confusion matrices to GCS.

    Args:
        config: dict, user passed parameters.
    """
    if config["output"]["use_dataflow"]:
        print("Performing polygon Dataflow pipeline!")
        beam_polygon.call_beam_polygon_pipeline(config)
        print("Dataflow polygon pipeline complete!")
    else:
        polygon_obj = polygon.LocalPolygonDoFn(
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
            height_scale_factor=(
                config["polygon"]["image_height"] /
                config["polygon"]["effective_slide_height"]
            ),
            width_scale_factor=(
                config["polygon"]["image_width"] /
                config["polygon"]["effective_slide_width"]
            ),
            timeout=config["polygon"]["timeout"]
        )

        csv_rows = []
        for threshold in config["polygon"]["thresholds"]:
            polygon_obj.apply_threshold(threshold)
            for dilation in config["polygon"]["dilation_factors"]:
                print(
                    "threshold = {}, dilation = {}".format(
                        threshold, dilation
                    )
                )
                confusion_matrix_dict = (
                    polygon_obj.get_polygon_confusion_matrix_dict(
                        dilation_factor=dilation
                    )
                )
                row_generator = confusion_matrix.confusion_matrix_to_csv(
                    element=confusion_matrix_dict
                )
                rows = [x for x in row_generator]
                if rows:
                    csv_rows.append(rows[0])

        print("Saving confusion matrices to CSV")
        tf.io.write_file(
            filename="{}_polygon_confusion_matrix.csv".format(
                config["output"]["output_gcs_path"]
            ),
            contents="\n".join(csv_rows),
        )
