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


class TrainDynamicThreshold(
    train_post_reconstruction.TrainPostReconstruction
):
    """Class used for training dynamic threshold model.
    """
    def __init__(self):
        """Instantiate instance of `TrainDynamicThreshold`.
        """
        pass

    def batch_mahalanobis_distance(self, batch_matrix, mu):
        """Calculates Mahalanobis distance of batch.

        Args:
            batch_matrix: tensor, rank 2 batch of data of shape
                (batch_size * height * width, num_cols).
            mu: tensor, rank 1 vector of learned error distribution column
                means.

        Returns:
            Rank 1 tensor of squared mahalanobis distances of shape
                (batch_size * height * width,).
        """
        y = tf.matmul(a=batch_matrix - mu, b=self.error_distribution_sigma_linv)

        return tf.einsum("ij,ij->i", y, y)

    def train_dynamic_threshold_supervised(self, mahalanobis_distances):
        """Trains dynamic threshold supervised.

        Args:
            mahalanobis_distances: tensor, rank 1 vector of squared
                mahalanobis distances of shape (batch_size * height * width,).
        """
        # TODO: Fill in supervised way later.
        pass

    def train_dynamic_threshold_unsupervised(self, mahalanobis_distances):
        """Trains dynamic threshold unsupervised.

        Args:
            mahalanobis_distances: tensor, rank 1 vector of squared
                mahalanobis distances of shape (batch_size * height * width,).
        """
        self.network_objects["dynamic_threshold"].batch_calculate_data_stats(
            data=tf.expand_dims(input=mahalanobis_distances, axis=-1)
        )

    def train_dynamic_threshold(self, features):
        """Trains dynamic threshold.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss tensor.
        """
        errors = self.get_image_absolute_errors_2d(features)

        mahalanobis_distances = self.batch_mahalanobis_distance(
            batch_matrix=errors,
            mu=self.network_objects["error_distribution"].col_means_vector
        )

        if self.params["training"]["dynamic_threshold"]["use_supervised"]:
            self.train_dynamic_threshold_supervised(mahalanobis_distances)
        else:
            self.train_dynamic_threshold_unsupervised(mahalanobis_distances)

        return tf.zeros(shape=(), dtype=tf.float32)
