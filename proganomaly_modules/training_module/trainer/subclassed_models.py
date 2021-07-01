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
from . import vector_to_image_networks


class Decoder(tf.keras.Model, vector_to_image_networks.VectorToImageNetwork):
    """Decoder that decodes vector into an image through adversarial training.

    Attributes:
        vector_to_image_name: str, name of decoder.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        growth_idx: int, the current resolution growth index.
        block_idx: int, the current resolution block index.
        vector_to_image_input_layer: `Input` layer of every
            VectorToImageNetwork model.
        projection_dense_layer: `WeightScaledDense` layer used for projecting
            noise latent vectors into an image.
        vector_to_image_conv_layers: list, `Conv2D` layers.
        vector_to_image_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` layers.
        vector_to_image_weighted_sum_layer: `WeightedSum` layer used for
            combining growing and shrinking network paths during transition
            phases.
        to_rgb_conv_layers: list, `Conv2D` toRGB layers.
        vector_to_image_outputs_fn: unbound function, function that returns
            the vector to image outputs.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        layers_dict,
        growth_idx
    ):
        """Instantiates and builds decoder network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizer for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
                variables.
            name: str, name of `Decoder`.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            layers_dict: dict, dictionary of Keras layers.
            growth_idx: int, the current resolution growth index.
        """
        tf.keras.Model.__init__(self)

        # Set names.
        self.vector_to_image_name = "{}_{}".format(name, "decoder")

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        self.growth_idx = growth_idx
        self.block_idx = (self.growth_idx + 1) // 2

        self.vector_to_image_input_layer = (
            layers_dict["vector_to_image_input_layer"]
        )
        self.projection_dense_layer = layers_dict["projection_dense_layer"]
        self.vector_to_image_conv_layers = (
            layers_dict["vector_to_image_conv_layers"]
        )
        self.vector_to_image_leaky_relu_layers = (
            layers_dict["vector_to_image_leaky_relu_layers"]
        )
        self.vector_to_image_weighted_sum_layer = (
            layers_dict["vector_to_image_weighted_sum_layer"]
        )
        self.to_rgb_conv_layers = layers_dict["to_rgb_conv_layers"]

        if self.growth_idx == 0:
            self.vector_to_image_outputs_fn = (
                self._get_vector_to_image_base_model_outputs
            )
        elif self.growth_idx % 2 == 1:
            self.vector_to_image_outputs_fn = (
                self._get_vector_to_image_growth_transition_model_outputs
            )
        else:
            self.vector_to_image_outputs_fn = (
                self._get_vector_to_image_growth_stable_model_outputs
            )

        self.build(
            input_shape=(None, self.params["generator"]["latent_size"])
        )

    def call(self, inputs, training=False):
        """Overrides call method of tf.keras.Model for forward pass.

        Args:
            inputs: tensor, rank 2 tensor of shape (batch_size, latent_size).
            training: bool, whether model is in training mode.

        Returns:
            vector_to_image_outputs: tensor, rank 4 tensor of shape
                (batch_size, 4 * 2 ** block_idx, 4 * 2 ** block_idx, depth).
        """
        vector_to_image_outputs = self.vector_to_image_outputs_fn(
            inputs, training, self.block_idx
        )

        return vector_to_image_outputs


class Encoder(tf.keras.Model, image_to_vector_networks.ImageToVectorNetwork):
    """Encoder that encodes an image into vector through adversarial training.

    Attributes:
        image_to_vector_name: str, name of encoder.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        growth_idx: int, the current resolution growth index.
        block_idx: int, the current resolution block index.
        image_to_vector_input_layers: list, `Input` layers for each resolution
            of image.
        from_rgb_conv_layers: list, `Conv2D` fromRGB layers.
        from_rgb_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` fromRGB layers.
        image_to_vector_conv_layers: list, `Conv2D` layers.
        image_to_vector_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` layers.
        growing_downsample_layers: list, `AveragePooling2D` layers for growing
            branch.
        shrinking_downsample_layers: list, `AveragePooling2D` layers for
            shrinking branch.
        image_to_vector_weighted_sum_layer: `WeightedSum` layer used for
            combining growing and shrinking network paths during transition
            phases.
        minibatch_stddev_layer: `MiniBatchStdDev` layer, applies minibatch
            stddev to image to add an additional feature channel based on the
            sample.
        flatten_layer: `Flatten` layer, flattens image for logits layer.
        logits_layer: `Dense` layer, used for calculating logits.
        unet_encoder_activations: list, activations after each conv block of
            ImageToVector network.
        image_to_vector_outputs_fn: unbound function, function that returns
            the image to vector outputs.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        layers_dict,
        growth_idx,
        network_type
    ):
        """Instantiates and builds encoder generator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizer for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
                variables.
            name: str, name of `Encoder`.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            layers_dict: dict, dictionary of Keras layers.
            growth_idx: int, the current resolution growth index.
            network_type: str, the network type: discriminator or encoder.
        """
        tf.keras.Model.__init__(self)

        # Set names.
        self.image_to_vector_name = "{}_{}".format(name, "encoder")

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        self.growth_idx = growth_idx
        self.block_idx = (self.growth_idx + 1) // 2

        self.image_to_vector_input_layers = (
            layers_dict["image_to_vector_input_layers"]
        )

        self.image_to_vector_input_layers = (
            layers_dict["image_to_vector_input_layers"]
        )

        self.from_rgb_conv_layers = layers_dict["from_rgb_conv_layers"]

        self.from_rgb_leaky_relu_layers = (
            layers_dict["from_rgb_leaky_relu_layers"]
        )

        self.image_to_vector_conv_layers = (
            layers_dict["image_to_vector_conv_layers"]
        )

        self.image_to_vector_leaky_relu_layers = (
            layers_dict["image_to_vector_leaky_relu_layers"]
        )

        self.growing_downsample_layers = (
            layers_dict["growing_downsample_layers"]
        )

        self.shrinking_downsample_layers = (
            layers_dict["shrinking_downsample_layers"]
        )

        self.image_to_vector_weighted_sum_layer = (
            layers_dict["image_to_vector_weighted_sum_layer"]
        )

        self.minibatch_stddev_layer = layers_dict["minibatch_stddev_layer"]

        self.flatten_layer = layers_dict["flatten_layer"]

        self.logits_layer = layers_dict["logits_layer"]

        self.unet_encoder_activations = [None] * 9

        if self.growth_idx == 0:
            self.image_to_vector_outputs_fn = (
                self._get_image_to_vector_base_model_outputs
            )
        elif self.growth_idx % 2 == 1:
            self.image_to_vector_outputs_fn = (
                self._get_image_to_vector_growth_transition_model_outputs
            )
        else:
            self.image_to_vector_outputs_fn = (
                self._get_image_to_vector_growth_stable_model_outputs
            )

        height, width = self.params["generator"]["projection_dims"][0:2]
        self.build(
            input_shape=(
                None,
                height * 2 ** self.block_idx,
                width * 2 ** self.block_idx,
                self.params["training"]["reconstruction"]["image_depth"]
            )
        )

    def call(self, inputs, training=False):
        """Overrides call method of tf.keras.Model for forward pass.

        Args:
            inputs: tensor, rank 4 tensor of shape
                (batch_size, 4 * 2 ** block_idx, 4 * 2 ** block_idx, depth).
            training: bool, whether model is in training mode.

        Returns:
            image_to_vector_outputs: tensor, rank 2 tensor of shape
                (batch_size, latent_size).
        """
        image_to_vector_outputs = self.image_to_vector_outputs_fn(
            inputs, training, self.block_idx
        )

        return image_to_vector_outputs


class UNetGenerator(
    tf.keras.Model,
    image_to_vector_networks.ImageToVectorNetwork,
    vector_to_image_networks.VectorToImageNetwork
):
    """U-net generator that creates an image through adversarial training.

    Attributes:
        image_to_vector_name: str, name of encoder.
        vector_to_image_name: str, name of decoder.
        kernel_regularizer: `l1_l2_regularizer` object, regularizer for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        growth_idx: int, the current resolution growth index.
        block_idx: int, the current resolution block index.
        image_to_vector_input_layers: list, `Input` layers for each resolution
            of image.
        from_rgb_conv_layers: list, `Conv2D` fromRGB layers.
        from_rgb_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` fromRGB layers.
        image_to_vector_conv_layers: list, `Conv2D` layers.
        image_to_vector_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` layers.
        growing_downsample_layers: list, `AveragePooling2D` layers for growing
            branch.
        shrinking_downsample_layers: list, `AveragePooling2D` layers for
            shrinking branch.
        image_to_vector_weighted_sum_layer: `WeightedSum` layer used for
            combining growing and shrinking network paths during transition
            phases.
        minibatch_stddev_layer: `MiniBatchStdDev` layer, applies minibatch
            stddev to image to add an additional feature channel based on the
            sample.
        flatten_layer: `Flatten` layer, flattens image for logits layer.
        logits_layer: `Dense` layer, used for calculating logits.
        vector_to_image_input_layer: `Input` layer of every
            VectorToImageNetwork model.
        projection_dense_layer: `WeightScaledDense` layer used for projecting
            noise latent vectors into an image.
        vector_to_image_conv_layers: list, `Conv2D` layers.
        vector_to_image_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` layers.
        vector_to_image_weighted_sum_layer: `WeightedSum` layer used for
            combining growing and shrinking network paths during transition
            phases.
        to_rgb_conv_layers: list, `Conv2D` toRGB layers.
        unet_encoder_activations: list, activations after each conv block of
            ImageToVector network.
        image_to_vector_outputs_fn: unbound function, function that returns
            the image to vector outputs.
        vector_to_image_outputs_fn: unbound function, function that returns
            the vector to image outputs.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        layers_dict,
        growth_idx
    ):
        """Instantiates and builds U-net generator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizer for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
                variables.
            name: str, name of `UNetGenerator`.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            layers_dict: dict, dictionary of Keras layers.
            growth_idx: int, the current resolution growth index.
        """
        tf.keras.Model.__init__(self)

        # Set names.
        self.image_to_vector_name = "{}_{}".format(name, "encoder")
        self.vector_to_image_name = "{}_{}".format(name, "decoder")

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        self.growth_idx = growth_idx
        self.block_idx = (self.growth_idx + 1) // 2

        self.image_to_vector_input_layers = (
            layers_dict["image_to_vector_input_layers"]
        )

        self.image_to_vector_input_layers = (
            layers_dict["image_to_vector_input_layers"]
        )

        self.from_rgb_conv_layers = layers_dict["from_rgb_conv_layers"]

        self.from_rgb_leaky_relu_layers = (
            layers_dict["from_rgb_leaky_relu_layers"]
        )

        self.image_to_vector_conv_layers = (
            layers_dict["image_to_vector_conv_layers"]
        )

        self.image_to_vector_leaky_relu_layers = (
            layers_dict["image_to_vector_leaky_relu_layers"]
        )

        self.growing_downsample_layers = (
            layers_dict["growing_downsample_layers"]
        )

        self.shrinking_downsample_layers = (
            layers_dict["shrinking_downsample_layers"]
        )

        self.image_to_vector_weighted_sum_layer = (
            layers_dict["image_to_vector_weighted_sum_layer"]
        )

        self.minibatch_stddev_layer = layers_dict["minibatch_stddev_layer"]

        self.flatten_layer = layers_dict["flatten_layer"]

        self.logits_layer = layers_dict["logits_layer"]

        self.vector_to_image_input_layer = (
            layers_dict["vector_to_image_input_layer"]
        )
        self.projection_dense_layer = layers_dict["projection_dense_layer"]
        self.vector_to_image_conv_layers = (
            layers_dict["vector_to_image_conv_layers"]
        )
        self.vector_to_image_leaky_relu_layers = (
            layers_dict["vector_to_image_leaky_relu_layers"]
        )
        self.vector_to_image_weighted_sum_layer = (
            layers_dict["vector_to_image_weighted_sum_layer"]
        )
        self.to_rgb_conv_layers = layers_dict["to_rgb_conv_layers"]

        self.unet_encoder_activations = [None] * 9

        if self.growth_idx == 0:
            self.image_to_vector_outputs_fn = (
                self._get_image_to_vector_base_model_outputs
            )

            self.vector_to_image_outputs_fn = (
                self._get_vector_to_image_base_model_outputs
            )
        elif self.growth_idx % 2 == 1:
            self.image_to_vector_outputs_fn = (
                self._get_image_to_vector_growth_transition_model_outputs
            )

            self.vector_to_image_outputs_fn = (
                self._get_vector_to_image_growth_transition_model_outputs
            )
        else:
            self.image_to_vector_outputs_fn = (
                self._get_image_to_vector_growth_stable_model_outputs
            )

            self.vector_to_image_outputs_fn = (
                self._get_vector_to_image_growth_stable_model_outputs
            )

        height, width = self.params["generator"]["projection_dims"][0:2]
        self.build(
            input_shape=(
                None,
                height * 2 ** self.block_idx,
                width * 2 ** self.block_idx,
                self.params["training"]["reconstruction"]["image_depth"]
            )
        )

    def call_encoder_model(self, inputs, training):
        """Calls encoder model for forward pass.

        Args:
            inputs: tensor, rank 4 tensor of shape
                (batch_size, 4 * 2 ** block_idx, 4 * 2 ** block_idx, depth).
            training: bool, whether model is in training mode.

        Returns:
            image_to_vector_outputs: tensor, rank 2 tensor of shape
                (batch_size, latent_size).
        """
        image_to_vector_outputs = self.image_to_vector_outputs_fn(
            inputs, training, self.block_idx
        )

        return image_to_vector_outputs

    def call_decoder_model(self, inputs, training):
        """Calls decoder model for forward pass.

        Args:
            inputs: tensor, rank 2 tensor of shape (batch_size, latent_size).
            training: bool, whether model is in training mode.

        Returns:
            vector_to_image_outputs: tensor, rank 4 tensor of shape
                (batch_size, 4 * 2 ** block_idx, 4 * 2 ** block_idx, depth).
        """
        vector_to_image_outputs = self.vector_to_image_outputs_fn(
            inputs, training, self.block_idx
        )

        return vector_to_image_outputs

    def call(self, inputs, training=False):
        """Overrides call method of tf.keras.Model for forward pass.

        Args:
            inputs: tensor, rank 4 tensor of shape
                (batch_size, 4 * 2 ** block_idx, 4 * 2 ** block_idx, depth).
            training: bool, whether model is in training mode.

        Returns:
            image_to_vector_outputs: tensor, rank 2 tensor of shape
                (batch_size, latent_size).
            vector_to_image_outputs: tensor, rank 4 tensor of shape
                (batch_size, 4 * 2 ** block_idx, 4 * 2 ** block_idx, depth).
        """
        image_to_vector_outputs = self.call_encoder_model(
            inputs=inputs, training=training
        )

        if (
            self.params["generator"]["GANomaly"]["add_uniform_noise_to_z"] and
            training
        ):
            # shape = (batch_size, latent_size)
            image_to_vector_outputs = tf.math.add(
                x=image_to_vector_outputs,
                y=tf.random.uniform(
                    shape=tf.shape(input=image_to_vector_outputs),
                    dtype=tf.float32
                )
            )

        vector_to_image_outputs = self.call_decoder_model(
            inputs=image_to_vector_outputs, training=training
        )

        return image_to_vector_outputs, vector_to_image_outputs
