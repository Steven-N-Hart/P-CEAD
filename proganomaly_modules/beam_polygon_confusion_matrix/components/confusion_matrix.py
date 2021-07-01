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


def confusion_matrix_to_csv(element):
    """Formats confusion matrix to CSV.

    Args:
        element_dict: dict, dictionary of confusion matrix quantities.

    Yields:
        Comma-delimited string of confusion matrix quantities.
    """
    csv_columns = (
        "slide_name",
        "threshold",
        "dilation_factor",
        "num_polygons",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives"
    )
    if element:
        yield ",".join([str(element[k]) for k in csv_columns])
