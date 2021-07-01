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


class TrainPostReconstruction(object):
    """Class used for training post-reconstruction models.
    """
    def __init__(self):
        """Instantiate instance of `TrainPostReconstruction`.
        """
        pass

    def get_image_absolute_errors_berg(self, real_images):
        """Gets image absolute errors using berg architecture.

        Args:
            real_images: tensor, rank 4 image of shape
                (batch_size, height, width, depth).

        Returns:
            Rank 4 tensor of image absolute errors of shape
                (batch_size, height, width, depth).
        """
        generator_model = self.network_objects["generator"].models[self.growth_idx]

        # Can't use berg architecture without encoder.
        assert(self.params["encoder"]["create"])

        encoder_model = self.network_objects["encoder"].models[self.growth_idx]

        # E(x).
        encoded_logits = encoder_model(inputs=real_images, training=False)

        # G(E(x)).
        encoded_images = generator_model(
            inputs=encoded_logits, training=False
        )

        # |x - G(E(x))|
        image_abs_error = tf.abs(x=real_images - encoded_images)

        return image_abs_error

    def get_image_absolute_errors_ganomaly(self, real_images):
        """Gets image absolute errors using GANomaly architecture.

        Args:
            real_images: tensor, rank 4 image of shape
                (batch_size, height, width, depth).

        Returns:
            Rank 4 tensor of image absolute errors of shape
                (batch_size, height, width, depth).
        """
        generator_model = self.network_objects["generator"].models[self.growth_idx]

        # G(x) = Gd(Ge(x)).
        _, gen_encoded_images = generator_model(
            inputs=real_images, training=False
        )

        # |x - G(x)|
        image_abs_error = tf.abs(x=real_images - gen_encoded_images)

        return image_abs_error

    def get_image_absolute_errors(self, real_images):
        """Gets image absolute errors.

        Args:
            real_images: tensor, rank 4 image of shape
                (batch_size, height, width, depth).

        Returns:
            Rank 4 tensor of image absolute errors of shape
                (batch_size, height, width, depth).
        """
        if self.params["generator"]["architecture"] == "berg":
            image_abs_error = self.get_image_absolute_errors_berg(real_images)
        elif self.params["generator"]["architecture"] == "GANomaly":
            image_abs_error = self.get_image_absolute_errors_ganomaly(
                real_images
            )

        return image_abs_error

    def reshape_image_absolute_errors(self, errors):
        """Reshapes image absolute errors.

        Args:
            errors: tensor, rank 4 image of shape
                (batch_size, height, width, depth).

        Returns:
            Rank 2 tensor of image absolute errors of shape
                (batch_size * height * width, depth).
        """
        errors_reshaped = tf.reshape(
            tensor=errors, shape=(-1, errors.shape[3])
        )

        return errors_reshaped

    def get_image_absolute_errors_2d(self, features):
        """Gets 2D image absolute errors.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Rank 2 tensor of image absolute errors of shape
                (batch_size * height * width, depth).
        """
        # Extract real images from features dictionary.
        real_images = self.resize_real_images(images=features["image"])

        errors = self.get_image_absolute_errors(real_images)
        errors_reshaped = self.reshape_image_absolute_errors(errors)

        return errors_reshaped
