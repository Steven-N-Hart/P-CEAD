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
from . import vector_to_image_networks


class GeneratorsFunctional(
    image_to_vector_networks.ImageToVectorNetwork,
    vector_to_image_networks.VectorToImageNetwork
):
    """Generator that creates an image through adversarial training.

    Attributes:
        network_type: str, the network type: decoder or unet.
        name: str, name of `Generator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        unet_encoder_activations: list, activations after each conv block of
            ImageToVector network.
        image_to_vector_models: list, instances of ImageToVectorNetwork
            `Model`s for each growth.
        vector_to_image_models: list, instances of VectorToImageNetwork
            `Model`s for each growth.
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

        if network_type == "unet":
            image_to_vector_networks.ImageToVectorNetwork.__init__(
                self,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_{}".format(self.name, "encoder"),
                params=self.params,
                alpha_var=self.alpha_var,
                network_type="encoder"
            )

        vector_to_image_networks.VectorToImageNetwork.__init__(
            self,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="{}_{}".format(self.name, "decoder"),
            params=self.params,
            alpha_var=self.alpha_var
        )

        self.unet_encoder_activations = [None] * 9

        # Store list of generator models.
        self.image_to_vector_models = []
        self.vector_to_image_models = []
        if network_type == "decoder":
            self.models = self._create_decoder_generator_models(num_growths)
        elif network_type == "unet":
            self.models = self._create_unet_generator_models(num_growths)
        else:
            self.models = []

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_decoder_generator_models(self, num_growths):
        """Creates list of decoder vector-to-image `Model`s for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `VectorToImageNetwork` `Model` objects.
        """
        models = []
        for growth_idx in range(num_growths):
            block_idx = (growth_idx + 1) // 2

            # Decoder model.
            decoder_inputs = self.vector_to_image_input_layer

            if growth_idx == 0:
                vector_to_image_outputs = (
                    self._get_vector_to_image_base_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )
            elif growth_idx % 2 == 1:
                vector_to_image_outputs = (
                    self._get_vector_to_image_growth_transition_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )
            elif growth_idx % 2 == 0:
                vector_to_image_outputs = (
                    self._get_vector_to_image_growth_stable_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )

            decoder_model = tf.keras.Model(
                inputs=decoder_inputs,
                outputs=vector_to_image_outputs,
                name="generator_decoder_{}".format(growth_idx)
            )
            self.vector_to_image_models.append(decoder_model)

            models.append(decoder_model)

        return models

    def _create_unet_generator_models(self, num_growths):
        """Creates list of U-net generator `Model` objects for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of U-net encoder-decoder `Model` objects.
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
                name="generator_unet_encoder_{}".format(growth_idx)
            )
            self.image_to_vector_models.append(encoder_model)

            # Decoder model.
            decoder_inputs = self.vector_to_image_input_layer

            if growth_idx == 0:
                vector_to_image_outputs = (
                    self._get_vector_to_image_base_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )
            elif growth_idx % 2 == 1:
                vector_to_image_outputs = (
                    self._get_vector_to_image_growth_transition_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )
            elif growth_idx % 2 == 0:
                vector_to_image_outputs = (
                    self._get_vector_to_image_growth_stable_model_outputs(
                        inputs=None, training=True, block_idx=block_idx
                    )
                )

            if (self.params["generator"]["architecture"] == "GANomaly" and
                any(self.params["generator"]["GANomaly"]["use_unet_skip_connections"])
               ):
                decoder_model = tf.keras.Model(
                    inputs=[encoder_inputs, decoder_inputs],
                    outputs=vector_to_image_outputs,
                    name="generator_unet_decoder_{}".format(growth_idx)
                )
            else:
                decoder_model = tf.keras.Model(
                    inputs=decoder_inputs,
                    outputs=vector_to_image_outputs,
                    name="unet_generator_decoder_{}".format(growth_idx)
                )
            self.vector_to_image_models.append(decoder_model)

            if (self.params["generator"]["architecture"] == "GANomaly" and
                any(self.params["generator"]["GANomaly"]["use_unet_skip_connections"])
               ):
                unet_model = tf.keras.Model(
                    inputs=encoder_inputs,
                    outputs=[
                        encoder_model(inputs=encoder_inputs),
                        decoder_model(
                            inputs=[
                                encoder_inputs,
                                encoder_model(inputs=encoder_inputs)
                            ]
                        )
                    ],
                    name="unet_generator_{}".format(growth_idx)
                )
            else:
                unet_model = tf.keras.Model(
                    inputs=encoder_inputs,
                    outputs=[
                        encoder_model(inputs=encoder_inputs),
                        decoder_model(
                            inputs=encoder_model(inputs=encoder_inputs)
                        )
                    ],
                    name="unet_generator_{}".format(growth_idx)
                )
            models.append(unet_model)

        return models


class GeneratorsSubClass(
    image_to_vector_networks.ImageToVectorNetwork,
    vector_to_image_networks.VectorToImageNetwork
):
    """Generators that creates an image through adversarial training.

    Attributes:
        network_type: str, the network type: decoder or unet.
        name: str, name of `Generator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        unet_encoder_activations: list, activations after each conv block of
            ImageToVector network.
        image_to_vector_models: list, instances of ImageToVectorNetwork
            `Model`s for each growth.
        vector_to_image_models: list, instances of VectorToImageNetwork
            `Model`s for each growth.
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

        vector_to_image_networks.VectorToImageNetwork.__init__(
            self,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="{}_{}".format(name, "decoder"),
            params=self.params,
            alpha_var=self.alpha_var
        )

        self.layers_dict = {
            "vector_to_image_input_layer": self.vector_to_image_input_layer,
            "projection_dense_layer": self.projection_dense_layer,
            "vector_to_image_conv_layers": self.vector_to_image_conv_layers,
            "vector_to_image_leaky_relu_layers": (
                self.vector_to_image_leaky_relu_layers
            ),
            "vector_to_image_weighted_sum_layer": (
                self.vector_to_image_weighted_sum_layer
            ),
            "to_rgb_conv_layers": self.to_rgb_conv_layers
        }

        if network_type == "decoder":
            self.models = self._create_decoder_generator_models(num_growths)
        elif network_type == "unet":
            image_to_vector_networks.ImageToVectorNetwork.__init__(
                self,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_{}".format(name, "encoder"),
                params=self.params,
                alpha_var=self.alpha_var,
                network_type="encoder"
            )

            self.layers_dict.update(
                {
                    "image_to_vector_input_layers": (
                        self.image_to_vector_input_layers
                    ),
                    "from_rgb_conv_layers": self.from_rgb_conv_layers,
                    "from_rgb_leaky_relu_layers": (
                        self.from_rgb_leaky_relu_layers
                    ),
                    "image_to_vector_conv_layers": (
                        self.image_to_vector_conv_layers
                    ),
                    "image_to_vector_leaky_relu_layers": (
                        self.image_to_vector_leaky_relu_layers
                    ),
                    "growing_downsample_layers": (
                        self.growing_downsample_layers
                    ),
                    "shrinking_downsample_layers": (
                        self.shrinking_downsample_layers
                    ),
                    "image_to_vector_weighted_sum_layer": (
                        self.image_to_vector_weighted_sum_layer
                    ),
                    "minibatch_stddev_layer": self.minibatch_stddev_layer,
                    "flatten_layer": self.flatten_layer,
                    "logits_layer": self.logits_layer
                }
            )

            self.models = self._create_unet_generator_models(num_growths)
        else:
            self.models = []

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_decoder_generator_models(self, num_growths):
        """Creates list of decoder vector-to-image `Model`s for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `Decoder` `Model` objects.
        """
        models = []
        for i in range(num_growths):
            generator = subclassed_models.Decoder(
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=self.name,
                params=self.params,
                alpha_var=self.alpha_var,
                layers_dict=self.layers_dict,
                growth_idx=i
            )

            models.append(generator)

        return models

    def _create_unet_generator_models(self, num_growths):
        """Creates list of U-net generator `Model` objects for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `UNetGenerator` `Model` objects.
        """
        models = []
        for i in range(num_growths):
            generator = subclassed_models.UNetGenerator(
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=self.name,
                params=self.params,
                alpha_var=self.alpha_var,
                layers_dict=self.layers_dict,
                growth_idx=i
            )

            models.append(generator)

        return models
