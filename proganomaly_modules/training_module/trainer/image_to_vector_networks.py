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

from . import custom_layers


class ImageToVectorNetwork(object):
    """Image-to-vector network with image input and outputs logits vector.

    Attributes:
        image_to_vector_network_type: str, the network type: discriminator or
            encoder.
        image_to_vector_name: str, name of `ImageToVectorNetwork`.
        image_to_vector_kernel_regularizer: `l1_l2_regularizer` object,
            regularizer for kernel variables.
        image_to_vector_bias_regularizer: `l1_l2_regularizer` object,
            regularizer for bias variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
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
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        network_type
    ):
        """Instantiates and builds image-to-vector network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of ImageToVector network.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            network_type: str, the network type: discriminator or encoder.
        """
        # Set whether it is a discriminator or encoder.
        self.image_to_vector_network_type = network_type

        # Set name of ImageToVector network.
        self.image_to_vector_name = name

        # Store regularizers.
        self.image_to_vector_kernel_regularizer = kernel_regularizer
        self.image_to_vector_bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        # Store lists of layers.
        self.image_to_vector_input_layers = []

        self.from_rgb_conv_layers = []
        self.from_rgb_leaky_relu_layers = []

        self.image_to_vector_conv_layers = []
        self.image_to_vector_leaky_relu_layers = []

        self.growing_downsample_layers = []
        self.shrinking_downsample_layers = []

        self.image_to_vector_weighted_sum_layer = None

        self.minibatch_stddev_layer = None

        self.flatten_layer = None
        self.logits_layer = None

        # Instantiate image_to_vector layers.
        self._create_image_to_vector_layers()

        self.unet_encoder_activations = [None] * 9

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_image_to_vector_input_layers(self):
        """Creates image_to_vector input layers for each image resolution.

        Returns:
            List of `Input` layers.
        """
        height, width = self.params["generator"]["projection_dims"][0:2]

        # Create list to hold `Input` layers.
        input_layers = [
            tf.keras.Input(
                shape=(
                    height * 2 ** i,
                    width * 2 ** i,
                    self.params["training"]["reconstruction"]["image_depth"]
                ),
                name="{}_{}x{}_inputs".format(
                    self.image_to_vector_name, height * 2 ** i, width * 2 ** i
                )
            )
            for i in range(len(self.params["discriminator"]["from_rgb_layers"]))
        ]

        return input_layers

    def _create_image_to_vector_from_rgb_layers(self):
        """Creates image_to_vector fromRGB layers of 1x1 convs.

        Returns:
            List of fromRGB 1x1 conv layers and leaky relu layers.
        """
        # Get fromRGB layer properties.
        from_rgb = [
            self.params["discriminator"]["from_rgb_layers"][i][0][:]
            for i in range(
                len(self.params["discriminator"]["from_rgb_layers"])
            )
        ]

        # Create list to hold toRGB 1x1 convs.
        from_rgb_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=from_rgb[i][3],
                kernel_size=from_rgb[i][0:2],
                strides=from_rgb[i][4:6],
                padding="same",
                activation=None,
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.image_to_vector_kernel_regularizer,
                bias_regularizer=self.image_to_vector_bias_regularizer,
                use_equalized_learning_rate=(
                    self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                ),
                name="{}_from_rgb_layers_conv2d_{}_{}x{}_{}_{}".format(
                    self.image_to_vector_name,
                    i,
                    from_rgb[i][0],
                    from_rgb[i][1],
                    from_rgb[i][2],
                    from_rgb[i][3]
                )
            )
            for i in range(len(from_rgb))
        ]

        from_rgb_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params[
                    "{}".format(self.image_to_vector_network_type)
                ]["leaky_relu_alpha"],
                name="{}_from_rgb_layers_leaky_relu_{}".format(
                    self.image_to_vector_name, i
                )
            )
            for i in range(len(from_rgb))
        ]

        return from_rgb_conv_layers, from_rgb_leaky_relu_layers

    def _create_image_to_vector_base_conv_layer_block(self):
        """Creates image_to_vector base conv layer block.

        Returns:
            List of base block conv layers and list of leaky relu layers.
        """
        # Get conv block layer properties.
        conv_block = self.params["discriminator"]["base_conv_blocks"][0]

        # Create list of base conv layers.
        base_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=conv_block[i][3],
                kernel_size=conv_block[i][0:2],
                strides=conv_block[i][4:6],
                padding="same",
                activation=None,
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.image_to_vector_kernel_regularizer,
                bias_regularizer=self.image_to_vector_bias_regularizer,
                use_equalized_learning_rate=(
                    self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                ),
                name="{}_base_layers_conv2d_{}_{}x{}_{}_{}".format(
                    self.image_to_vector_name,
                    i,
                    conv_block[i][0],
                    conv_block[i][1],
                    conv_block[i][2],
                    conv_block[i][3]
                )
            )
            for i in range(len(conv_block))
        ]

        base_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params[
                    "{}".format(self.image_to_vector_network_type)
                ]["leaky_relu_alpha"],
                name="{}_base_layers_leaky_relu_{}".format(
                    self.image_to_vector_name, i
                )
            )
            for i in range(len(conv_block))
        ]

        return base_conv_layers, base_leaky_relu_layers

    def _create_image_to_vector_growth_conv_layer_block(self, block_idx):
        """Creates image_to_vector growth conv layer block.

        Args:
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block's conv layers and list of growth block's
                leaky relu layers.
        """
        # Get conv block layer properties.
        conv_block = (
            self.params["discriminator"]["growth_conv_blocks"][block_idx]
        )

        # Create new growth convolutional layers.
        growth_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=conv_block[i][3],
                kernel_size=conv_block[i][0:2],
                strides=conv_block[i][4:6],
                padding="same",
                activation=None,
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.image_to_vector_kernel_regularizer,
                bias_regularizer=self.image_to_vector_bias_regularizer,
                use_equalized_learning_rate=(
                    self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                ),
                name="{}_growth_layers_conv2d_{}_{}_{}x{}_{}_{}".format(
                    self.image_to_vector_name,
                    block_idx,
                    i,
                    conv_block[i][0],
                    conv_block[i][1],
                    conv_block[i][2],
                    conv_block[i][3]
                )
            )
            for i in range(len(conv_block))
        ]

        growth_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params[
                    "{}".format(self.image_to_vector_network_type)
                ]["leaky_relu_alpha"],
                name="{}_growth_layers_leaky_relu_{}_{}".format(
                    self.image_to_vector_name, block_idx, i
                )
            )
            for i in range(len(conv_block))
        ]

        return growth_conv_layers, growth_leaky_relu_layers

    def _create_downsample_layers(self):
        """Creates image_to_vector downsample layers.

        Returns:
            Lists of AveragePooling2D layers for growing and shrinking
                branches.
        """
        # Create list to hold growing branch's downsampling layers.
        growing_downsample_layers = [
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_growing_average_pooling_2d_{}".format(
                    self.image_to_vector_name, i - 1
                )
            )
            for i in range(
                1, len(self.params["discriminator"]["from_rgb_layers"])
            )
        ]

        # Create list to hold shrinking branch's downsampling layers.
        shrinking_downsample_layers = [
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_shrinking_average_pooling_2d_{}".format(
                    self.image_to_vector_name, i - 1
                )
            )
            for i in range(
                1, len(self.params["discriminator"]["from_rgb_layers"])
            )
        ]

        return growing_downsample_layers, shrinking_downsample_layers

    def _create_image_to_vector_layers(self):
        """Creates image_to_vector layers.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                (batch_size, latent_size).
        """
        # Create input layers for each image resolution.
        self.image_to_vector_input_layers = (
            self._create_image_to_vector_input_layers()
        )

        (self.from_rgb_conv_layers,
         self.from_rgb_leaky_relu_layers) = (
            self._create_image_to_vector_from_rgb_layers()
        )

        (base_conv_layers,
         base_leaky_relu_layers) = (
            self._create_image_to_vector_base_conv_layer_block()
        )
        self.image_to_vector_conv_layers.append(base_conv_layers)
        self.image_to_vector_leaky_relu_layers.append(base_leaky_relu_layers)

        for block_idx in range(
            len(self.params["discriminator"]["growth_conv_blocks"])
        ):
            (growth_conv_layers,
             growth_leaky_relu_layers
             ) = self._create_image_to_vector_growth_conv_layer_block(
                 block_idx
                )

            self.image_to_vector_conv_layers.append(growth_conv_layers)
            self.image_to_vector_leaky_relu_layers.append(
                growth_leaky_relu_layers
            )

        (self.growing_downsample_layers,
         self.shrinking_downsample_layers) = self._create_downsample_layers()

        self.image_to_vector_weighted_sum_layer = custom_layers.WeightedSum(
            alpha=self.alpha_var, name="weighted_sum_{}_{}".format(
                self.image_to_vector_network_type, self.image_to_vector_name
            )
        )

        self.minibatch_stddev_layer = custom_layers.MiniBatchStdDev(
            params={
                "use_minibatch_stddev": (
                    self.params[
                        "{}".format(self.image_to_vector_network_type)
                    ]["use_minibatch_stddev"]
                ),
                "group_size": (
                    self.params[
                        "{}".format(self.image_to_vector_network_type)
                    ]["minibatch_stddev_group_size"]
                ),
                "use_averaging": (
                    self.params[
                        "{}".format(self.image_to_vector_network_type)
                    ]["minibatch_stddev_use_averaging"]
                )
            },
            name="minibatch_stddev_{}".format(
                self.image_to_vector_network_type
            )
        )

        self.flatten_layer = tf.keras.layers.Flatten()

        self.logits_layer = custom_layers.WeightScaledDense(
            units=(
                1
                if self.image_to_vector_network_type == "discriminator"
                else self.params["generator"]["latent_size"]
            ),
            activation=None,
            kernel_initializer=(
                tf.random_normal_initializer(mean=0., stddev=1.0)
                if self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
                else "he_normal"
            ),
            kernel_regularizer=self.image_to_vector_kernel_regularizer,
            bias_regularizer=self.image_to_vector_bias_regularizer,
            use_equalized_learning_rate=(
                self.params["training"]["reconstruction"]["use_equalized_learning_rate"]
            ),
            name="{}_layers_dense_logits".format(self.image_to_vector_name)
        )

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _use_image_to_vector_logits_layer(self, inputs):
        """Uses flatten and logits layers to get logits tensor.

        Args:
            inputs: tensor, output of last conv layer of image_to_vector.

        Returns:
            Final logits tensor of image_to_vector.
        """
        # Set shape to remove ambiguity for dense layer.
        height, width = self.params["generator"]["projection_dims"][0:2]
        valid_kernel_size = (
            self.params["discriminator"]["base_conv_blocks"][0][-1][0]
        )
        inputs.set_shape(
            (
                inputs.get_shape()[0],
                height - valid_kernel_size + 1,
                width - valid_kernel_size + 1,
                inputs.get_shape()[-1]
            )
        )

        # Flatten final block conv tensor.
        flat_inputs = self.flatten_layer(inputs=inputs)

        # Final linear layer for logits.
        logits = self.logits_layer(inputs=flat_inputs)

        return logits

    def _create_image_to_vector_base_block_and_logits(self, inputs):
        """Creates base image_to_vector block and logits.

        Args:
            inputs: tensor, output of previous `Conv2D` block's layer.

        Returns:
            Final logits tensor of image_to_vector.
        """
        # Only need the first conv layer block for base network.
        base_conv_layers = self.image_to_vector_conv_layers[0]
        base_leaky_relu_layers = self.image_to_vector_leaky_relu_layers[0]

        network = self.minibatch_stddev_layer(inputs=inputs)
        for i in range(len(base_conv_layers)):
            network = base_conv_layers[i](inputs=network)
            network = base_leaky_relu_layers[i](inputs=network)

        if (
            self.params["generator"]["architecture"] == "GANomaly" and
            self.params["generator"]["GANomaly"]["use_unet_skip_connections"][0]
        ):
            self.unet_encoder_activations[0] = network

        # Have valid padding for layer just before flatten and logits.
        # i.e. (batch_size, 4, 4, 512) -> (batch_size, 1, 1, 512)
        network = network[:, :1, :1, :]

        # Get logits now.
        logits = self._use_image_to_vector_logits_layer(inputs=network)

        return logits

    def _create_image_to_vector_growth_transition_weighted_sum(
        self, inputs, block_idx
    ):
        """Creates growth transition img_to_vec weighted_sum.

        Args:
            inputs: tensor, input image to image_to_vector.
            block_idx: int, current block index of model progression.

        Returns:
            Tensor of weighted sum between shrinking and growing block paths.
        """
        # Growing side chain.
        growing_from_rgb_conv_layer = self.from_rgb_conv_layers[block_idx]
        growing_from_rgb_leaky_relu_layer = (
            self.from_rgb_leaky_relu_layers[block_idx]
        )
        growing_downsample_layer = (
            self.growing_downsample_layers[block_idx - 1]
        )

        growing_conv_layers = self.image_to_vector_conv_layers[block_idx]
        growing_leaky_relu_layers = (
            self.image_to_vector_leaky_relu_layers[block_idx]
        )

        # Pass inputs through layer chain.
        network = growing_from_rgb_conv_layer(inputs=inputs)
        network = growing_from_rgb_leaky_relu_layer(inputs=network)

        for i in range(len(growing_conv_layers)):
            network = growing_conv_layers[i](inputs=network)
            network = growing_leaky_relu_layers[i](inputs=network)

        if (
            self.params["generator"]["architecture"] == "GANomaly" and
            self.params["generator"]["GANomaly"]["use_unet_skip_connections"][block_idx]
        ):
            self.unet_encoder_activations[block_idx] = network

        # Down sample from 2s X 2s to s X s image.
        growing_network = growing_downsample_layer(inputs=network)

        # Shrinking side chain.
        shrinking_from_rgb_conv_layer = (
            self.from_rgb_conv_layers[block_idx - 1]
        )
        shrinking_from_rgb_leaky_relu_layer = (
            self.from_rgb_leaky_relu_layers[block_idx - 1]
        )
        shrinking_downsample_layer = (
            self.shrinking_downsample_layers[block_idx - 1]
        )

        # Pass inputs through layer chain.
        # Down sample from 2s X 2s to s X s image.
        network = shrinking_downsample_layer(inputs=inputs)

        network = shrinking_from_rgb_conv_layer(inputs=network)
        shrinking_network = shrinking_from_rgb_leaky_relu_layer(
            inputs=network
        )

        # Weighted sum.
        weighted_sum = self.image_to_vector_weighted_sum_layer(
            inputs=[growing_network, shrinking_network]
        )

        return weighted_sum

    def _create_image_to_vector_perm_growth_block_network(
        self, inputs, block_idx, stable
    ):
        """Creates image_to_vector permanent block network.

        Args:
            inputs: tensor, output of previous block's layer.
            block_idx: int, current block index of model progression.
            stable: int, integer flag if being used in a stable model (1) or
                not (0).

        Returns:
            Tensor from final permanent block `Conv2D` layer.
        """
        gen_params = self.params["generator"]
        # Get permanent growth blocks, so skip the base block.
        permanent_conv_layers = (
            self.image_to_vector_conv_layers[1:block_idx + stable]
        )
        permanent_leaky_relu_layers = (
            self.image_to_vector_leaky_relu_layers[1:block_idx + stable]
        )
        permanent_downsample_layers = (
            self.growing_downsample_layers[0:block_idx + stable - 1]
        )

        # Reverse order of blocks.
        permanent_conv_layers = permanent_conv_layers[::-1]
        permanent_leaky_relu_layers = permanent_leaky_relu_layers[::-1]
        permanent_downsample_layers = permanent_downsample_layers[::-1]

        # Pass inputs through layer chain.
        network = inputs

        # Loop through the permanent growth blocks.
        for i in range(len(permanent_conv_layers)):
            # Get layers from ith permanent block.
            conv_layers = permanent_conv_layers[i]
            leaky_relu_layers = permanent_leaky_relu_layers[i]
            permanent_downsample_layer = permanent_downsample_layers[i]

            # Loop through layers of ith permanent block.
            for j in range(len(conv_layers)):
                network = conv_layers[j](inputs=network)
                network = leaky_relu_layers[j](inputs=network)

            if gen_params["architecture"] == "GANomaly":
                skip_idx = block_idx + stable - 1 - i
                if gen_params["GANomaly"]["use_unet_skip_connections"][skip_idx]:
                    self.unet_encoder_activations[skip_idx] = network

            # Down sample from 2s X 2s to s X s image.
            network = permanent_downsample_layer(inputs=network)

        return network

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _get_image_to_vector_base_model_outputs(self, inputs, training, block_idx):
        """Builds image_to_vector base model.

        Args:
            input_shape: tuple, shape of image vector input of shape
                (batch_size, height, width, depth).
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to image_to_vector.
        # shape = (batch_size, height, width, depth)
        if not self.params["training"]["subclass_models"]:
            inputs = self.image_to_vector_input_layers[0]

        # Only need the first fromRGB conv layer & block for base network.
        base_from_rgb_conv_layer = self.from_rgb_conv_layers[0]
        base_from_rgb_leaky_relu_layer = self.from_rgb_leaky_relu_layers[0]

        base_conv_layers = self.image_to_vector_conv_layers[0]
        base_leaky_relu_layers = self.image_to_vector_leaky_relu_layers[0]

        # Pass inputs through layer chain.
        network = base_from_rgb_conv_layer(inputs=inputs)
        network = base_from_rgb_leaky_relu_layer(inputs=network)

        # Get logits after continuing through base conv block.
        logits = self._create_image_to_vector_base_block_and_logits(
            inputs=network
        )

        return logits

    def _get_image_to_vector_growth_transition_model_outputs(
        self, inputs, training, block_idx
    ):
        """Builds image_to_vector growth transition model.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                (batch_size, height, width, depth).
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to image_to_vector.
        # shape = (batch_size, height, width, depth)
        if not self.params["training"]["subclass_models"]:
            inputs = self.image_to_vector_input_layers[block_idx]

        # Get weighted sum between shrinking and growing block paths.
        weighted_sum = (
            self._create_image_to_vector_growth_transition_weighted_sum(
                inputs=inputs, block_idx=block_idx
            )
        )

        # Get output of final permanent growth block's last `Conv2D` layer.
        network = self._create_image_to_vector_perm_growth_block_network(
            inputs=weighted_sum, block_idx=block_idx, stable=0
        )

        # Get logits after continuing through base conv block.
        logits = self._create_image_to_vector_base_block_and_logits(
            inputs=network
        )

        return logits

    def _get_image_to_vector_growth_stable_model_outputs(
        self, inputs, training, block_idx
    ):
        """Builds image_to_vector growth stable model.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                (batch_size, latent_size).
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to image_to_vector.
        # shape = (batch_size, latent_size)
        if not self.params["training"]["subclass_models"]:
            inputs = self.image_to_vector_input_layers[block_idx]

        # Get fromRGB layers.
        from_rgb_conv_layer = self.from_rgb_conv_layers[block_idx]
        from_rgb_leaky_relu_layer = self.from_rgb_leaky_relu_layers[block_idx]

        # Pass inputs through layer chain.
        network = from_rgb_conv_layer(inputs=inputs)
        network = from_rgb_leaky_relu_layer(inputs=network)

        # Get output of final permanent growth block's last `Conv2D` layer.
        network = self._create_image_to_vector_perm_growth_block_network(
            inputs=network, block_idx=block_idx, stable=1
        )

        # Get logits after continuing through base conv block.
        logits = self._create_image_to_vector_base_block_and_logits(
            inputs=network
        )

        return logits
