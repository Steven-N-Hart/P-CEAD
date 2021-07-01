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


class Anomaly(object):
    """Class used for anomaly detection & localization.
    """
    def __init__(self):
        """Instantiate instance of `Anomaly`.
        """
        pass

    def minmax_normalization(self, X):
        """Min-max normalizes images.

        Args:
            X: tensor, image tensor of rank 4.

        Returns:
            Min-max normalized image tensor with same shape as X.
        """
        # Keep dims for broadcasting.
        min_x = tf.reduce_min(
            input_tensor=X,
            axis=(1, 2, 3),
            keepdims=True,
            name="minmax_normalization_min"
        )
        max_x = tf.reduce_max(
            input_tensor=X,
            axis=(1, 2, 3),
            keepdims=True,
            name="minmax_normalization_max"
        )

        X_normalized = tf.math.divide_no_nan(
            x=X - min_x,
            y=max_x - min_x,
            name="minmax_normalization_normalized"
        )

        return X_normalized

    def get_residual_loss(self, query_images, encoded_images):
        """Gets residual loss between query and encoded images.

        Args:
            query_images: tensor, query image input for predictions.
            encoded_images: tensor, image from generator from encoder's logits.

        Returns:
            Residual loss scalar tensor.
        """
        # Minmax normalize query images.
        query_images_normalized = self.minmax_normalization(X=query_images)

        # Minmax normalize encoded images.
        encoded_images_normalized = self.minmax_normalization(X=encoded_images)

        # Find pixel difference between normalized query and encoded images.
        image_difference = tf.subtract(
            x=query_images_normalized,
            y=encoded_images_normalized,
            name="image_difference"
        )

        # Take L2 norm of difference.
        image_difference_l2_norm = tf.reduce_sum(
            input_tensor=tf.square(x=image_difference),
            axis=[1, 2, 3],
            name="image_difference_l2_norm"
        )

        # Scale by image dimension sizes to get residual loss.
        height, width = self.params["generator"]["projection_dims"][0:2]
        height *= 2 ** self.block_idx
        width *= 2 ** self.block_idx
        depth = self.params["training"]["reconstruction"]["image_depth"]
        residual_loss = tf.divide(
            x=image_difference_l2_norm,
            y=tf.cast(
                x=height * width * depth, dtype=tf.float32
            ),
            name="residual_loss"
        )

        return residual_loss

    def get_origin_distance_loss(self, encoded_logits):
        """Gets origin distance loss measuring distance of z-hat from origin.

        Args:
            encoded_logits: tensor, encoder's logits encoded from query images.

        Returns:
            Origin distance loss scalar tensor.
        """
        # Take L2 norm z-hat.
        z_hat_l2_norm = tf.sqrt(
            x=tf.reduce_sum(
                input_tensor=tf.square(x=encoded_logits),
                axis=-1
            ) + 1e-8,
            name="z_hat_l2_norm"
        )

        # Scale by latent size to get origin distance loss.
        origin_distance_loss = tf.divide(
            x=-z_hat_l2_norm,
            y=tf.sqrt(
                x=tf.cast(
                    x=self.params["generator"]["latent_size"], dtype=tf.float32
                )
            ),
            name="origin_distance_loss"
        )

        return origin_distance_loss

    def get_anomaly_scores(
        self, query_images, encoded_logits, encoded_images
    ):
        """Gets anomaly scores from query and encoded images.

        Args:
            query_images: tensor, query image input for predictions.
            encoded_logits: tensor, encoder's logits encoded from query images.
            encoded_images: tensor, image from generator from encoder's logits.

        Returns:
            Predictions dictionary and export outputs dictionary.
        """
        # Get residual loss.
        residual_loss = self.get_residual_loss(query_images, encoded_images)

        # Get origin distance loss.
        origin_dist_loss = self.get_origin_distance_loss(encoded_logits)

        # Get anomaly scores.
        residual_scl = self.params["export"]["anom_convex_combo_factor"] * residual_loss
        one_minus_factor = 1. - self.params["export"]["anom_convex_combo_factor"]
        origin_scl = one_minus_factor * origin_dist_loss
        anomaly_scores = tf.add(
            x=residual_scl, y=origin_scl, name="anomaly_scores"
        )

        return anomaly_scores

    def anomaly_detection(self, query_images, encoded_logits, encoded_images):
        """Gets anomaly scores and flags from query and encoded images.

        Args:
            query_images: tensor, query image input for predictions.
            encoded_logits: tensor, encoder's logits encoded from query images.
            encoded_images: tensor, image from generator from encoder's logits.

        Returns:
            Anomaly scores tensor of shape (batch_size,) and anomaly flags
                tensor of shape (batch_size,).
        """
        # Get anomaly scores.
        anomaly_scores = self.get_anomaly_scores(
            query_images, encoded_logits, encoded_images
        )

        # Get anomaly flags.
        anomaly_flags = tf.cast(
            x=tf.greater(
                x=anomaly_scores, 
                y=self.params["export"]["anomaly_threshold"]
            ),
            dtype=tf.int32,
            name="anomaly_flags"
        )

        return anomaly_scores, anomaly_flags

    def zero_center_sigmoid_absolute_values(self, absolute_values):
        """Centers absolute values from range [0, inf) to [-1, 1).

        Args:
            absolute_values: tensor, absolute values with range [0, inf).

        Returns:
            Tensor with range [-1, 1).
        """
        # Apply sigmoid to smoothly squash between [0, 1].
        # However since the abs limits the domain to [0, inf),
        # the range will actually be [0.5, 1).
        sigmoid_absolute_values = tf.math.sigmoid(x=absolute_values)

        # Scale images to [-1, 1).
        zero_centered = sigmoid_absolute_values * 4. - 3.

        return zero_centered

    def anomaly_localization_sigmoid(self, query_images, encoded_images):
        """Gets pixel-level anomaly scores between query and encoded images.

        Args:
            query_images: tensor, query image input for predictions.
            encoded_images: tensor, image from generator from encoder's
                logits.

        Returns:
            Pixel-level anomaly scores tensor of shape
                (batch_size, image_height, image_width, depth).
        """
        # Get pixel-level anomaly scores.
        pixel_level_anomaly_scores = tf.abs(query_images - encoded_images)

        # Scale range from [0, inf) to [-1, 1).
        anomaly_images = self.zero_center_sigmoid_absolute_values(
            absolute_values=pixel_level_anomaly_scores
        )

        return anomaly_images

    def anomaly_localization_linear(self, query_images, encoded_images):
        """Gets pixel-level anomaly scores between query and encoded images.

        Args:
            query_images: tensor, query image input for predictions.
            encoded_images: tensor, image from generator from encoder's
                logits.

        Returns:
            Pixel-level anomaly scores tensor of shape
                (batch_size, image_height, image_width, depth).
        """
        # Get pixel-level anomaly scores.
        pixel_level_anomaly_scores = tf.abs(query_images - encoded_images)

        # Min-max normalize scores to scale range to [0, 1].
        normalized_scores = self.minmax_normalization(
            X=pixel_level_anomaly_scores
        )

        # Scale images to [-1, 1).
        anomaly_images = normalized_scores * 2. - 1.

        return anomaly_images
