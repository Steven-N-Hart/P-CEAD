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

import json


def segmentation_dict_to_json(element_dict, key):
    """Formats segmentation dict to json.

    Args:
        element_dict: dict, dictionary of inference outputs.

    Yields:
        JSON serialized string.
    """
    segmentation_dict = element_dict[key]
    for k, v in segmentation_dict.items():
        yield json.dumps({k: v})
