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

from . import image_to_vector_networks
from . import subclassed_models


class DiscriminatorsFunctional(image_to_vector_networks.ImageToVectorNetwork):
    """Discriminator that tries to discern generated from real images.

    Attributes:
        network_type: str, the network type: discriminator or encoder.
        name: str, name of `Discriminator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        unet_encoder_activations: list, activations after each conv block of
            ImageToVector network.
        models: list, instances of `Model` for each growth.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        num_growths,
        network_type
    ):
        """Instantiates and builds discriminator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizer for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
                variables.
            name: str, name of generator.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            num_growths: int, number of growth phases for model.
            network_type: str, the network type: decoder or unet.
        """
        # Set whether it is a discriminator or encoder.
        self.network_type = network_type

        # Set name of generator.
        self.name = name

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        image_to_vector_networks.ImageToVectorNetwork.__init__(
            self,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name=self.name,
            params=self.params,
            alpha_var=self.alpha_var,
            network_type=network_type
        )

        self.unet_encoder_activations = [None] * 9

        # Store list of generator models.
        self.models = self._create_encoder_discriminator_models(num_growths)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_encoder_discriminator_models(self, num_growths):
        """Creates list of image_to_vector `Model`s for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `ImageToVectorNetwork` `Model` objects.
        """
        models = []
        for growth_idx in range(num_growths):
            block_idx = (growth_idx + 1) // 2

            # Encoder model.
            encoder_inputs = self.image_to_vector_input_layers[block_idx]

            if growth_idx == 0:
                image_to_vector_model_outputs = (
                    self._get_image_to_vector_base_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )
            elif growth_idx % 2 == 1:
                image_to_vector_model_outputs = (
                    self._get_image_to_vector_growth_transition_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )
            elif growth_idx % 2 == 0:
                image_to_vector_model_outputs = (
                    self._get_image_to_vector_growth_stable_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )

            encoder_model = tf.keras.Model(
                inputs=encoder_inputs,
                outputs=image_to_vector_model_outputs,
                name="{}_{}".format(self.network_type, growth_idx)
            )
            models.append(encoder_model)

        return models


class DiscriminatorsSubClass(image_to_vector_networks.ImageToVectorNetwork):
    """Discriminator that tries to discern generated from real images.

    Attributes:
        network_type: str, the network type: discriminator or encoder.
        name: str, name of `Discriminator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        unet_encoder_activations: list, activations after each conv block of
            ImageToVector network.
        models: list, instances of `Model` for each growth.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        num_growths,
        network_type
    ):
        """Instantiates and builds generator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizer for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
                variables.
            name: str, name of generator.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            num_growths: int, number of growth phases for model.
            network_type: str, the network type: decoder or unet.
        """
        # Set whether it is a discriminator or encoder.
        self.network_type = network_type

        # Set name of generator.
        self.name = name

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        image_to_vector_networks.ImageToVectorNetwork.__init__(
            self,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name=self.name,
            params=self.params,
            alpha_var=self.alpha_var,
            network_type=network_type
        )

        self.layers_dict = {
            "image_to_vector_input_layers": self.image_to_vector_input_layers,
            "from_rgb_conv_layers": self.from_rgb_conv_layers,
            "from_rgb_leaky_relu_layers": self.from_rgb_leaky_relu_layers,
            "image_to_vector_conv_layers": self.image_to_vector_conv_layers,
            "image_to_vector_leaky_relu_layers": (
                self.image_to_vector_leaky_relu_layers
            ),
            "growing_downsample_layers": self.growing_downsample_layers,
            "shrinking_downsample_layers": self.shrinking_downsample_layers,
            "image_to_vector_weighted_sum_layer": (
                self.image_to_vector_weighted_sum_layer
            ),
            "minibatch_stddev_layer": self.minibatch_stddev_layer,
            "flatten_layer": self.flatten_layer,
            "logits_layer": self.logits_layer
        }

        # Store list of generator models.
        self.models = self._create_encoder_discriminator_models(num_growths)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_encoder_discriminator_models(self, num_growths):
        """Creates list of image_to_vector `Model`s for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `ImageToVectorNetwork` `Model` objects.
        """
        models = []
        for i in range(num_growths):
            encoder = subclassed_models.Encoder(
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=self.name,
                params=self.params,
                alpha_var=self.alpha_var,
                layers_dict=self.layers_dict,
                growth_idx=i,
                network_type=self.network_type
            )

            models.append(encoder)

        return models
