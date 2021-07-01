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


def pre_polygons(thresholds, dilation_factors):
    """Prepares polygon threshold and dilation parameters.

    Args:
        thresholds: list, thresholds to apply to images in range [0., 1.].
        dilation_factors: list, factors to dilate polygons in range [0., inf).

    Yields:
        2-tuple of threshold and dilation factor.
    """
    for threshold in thresholds:
        for dilation_factor in dilation_factors:
            yield (threshold, dilation_factor)
