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

from . import train_reconstruction
from . import train_error_distribution
from . import train_dynamic_threshold


class Train(
    train_reconstruction.TrainReconstruction,
    train_error_distribution.TrainErrorDistribution,
    train_dynamic_threshold.TrainDynamicThreshold
):
    """Class used for training various models.
    """
    def __init__(self):
        """Instantiate instance of `Train`.
        """
        pass
