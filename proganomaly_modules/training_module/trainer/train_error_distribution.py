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

from . import train_post_reconstruction


class TrainErrorDistribution(
    train_post_reconstruction.TrainPostReconstruction
):
    """Class used for training error distribution model.
    """
    def __init__(self):
        """Instantiate instance of `TrainErrorDistribution`.
        """
        pass

    def train_error_distribution(self, features):
        """Trains error distribution.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss tensor.
        """
        errors = self.get_image_absolute_errors_2d(features)

        self.network_objects["error_distribution"].batch_calculate_data_stats(
            data=errors
        )

        return tf.zeros(shape=(), dtype=tf.float32)
