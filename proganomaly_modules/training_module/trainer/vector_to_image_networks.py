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


class VectorToImageNetwork(object):
    """Vector-to-image network that takes vector input and outputs image.

    Attributes:
        vector_to_image_name: str, name of `VectorToImageNetwork`.
        vector_to_image_kernel_regularizer: `l1_l2_regularizer` object,
            regularizer for kernel variables.
        vector_to_image_bias_regularizer: `l1_l2_regularizer` object,
            regularizer for bias variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
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
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var
    ):
        """Instantiates and builds vector to image network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizer for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizer for bias
                variables.
            name: str, name of VectorToImageNetwork.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
        """
        # Set name of VectorToImageNetwork.
        self.vector_to_image_name = name

        # Store regularizers.
        self.vector_to_image_kernel_regularizer = kernel_regularizer
        self.vector_to_image_bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        # Store lists of layers.
        self.vector_to_image_input_layer = None
        self.projection_dense_layer = None
        self.vector_to_image_conv_layers = []
        self.vector_to_image_leaky_relu_layers = []
        self.vector_to_image_weighted_sum_layer = None
        self.to_rgb_conv_layers = []

        # Instantiate vector_to_image layers.
        self._create_vector_to_image_layers()

        self.unet_encoder_activations = [None] * 9

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_vector_to_image_input_layer(self):
        """Creates `Input` layer.

        Returns:
            Instance of `Input` layer.
        """
        return tf.keras.Input(
            shape=(self.params["generator"]["latent_size"],),
            name="{}_inputs".format(self.vector_to_image_name)
        )

    def _create_vector_to_image_projection_dense_layer(self):
        """Creates projection dense layer.

        Dense layer converts latent vectors to flattened images.

        Returns:
            Instance of `WeightScaledDense` layer.
        """
        recon_dict = self.params["training"]["reconstruction"]
        projection_height = self.params["generator"]["projection_dims"][0]
        projection_width = self.params["generator"]["projection_dims"][1]
        projection_depth = self.params["generator"]["projection_dims"][2]

        # shape = (
        #     batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        return custom_layers.WeightScaledDense(
            units=projection_height * projection_width * projection_depth,
            activation=None,
            kernel_initializer=(
                tf.random_normal_initializer(mean=0., stddev=1.0)
                if recon_dict["use_equalized_learning_rate"]
                else "he_normal"
            ),
            kernel_regularizer=self.vector_to_image_kernel_regularizer,
            bias_regularizer=self.vector_to_image_bias_regularizer,
            use_equalized_learning_rate=(
                recon_dict["use_equalized_learning_rate"]
            ),
            name="projection_dense_layer"
        )

    def _create_vector_to_image_base_conv_layer_block(self):
        """Creates vector_to_image base conv layer block.

        Returns:
            List of base block conv layers and list of leaky relu layers.
        """
        recon_dict = self.params["training"]["reconstruction"]
        # Get conv block layer properties.
        conv_block = self.params["generator"]["base_conv_blocks"][0]

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
                    if recon_dict["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.vector_to_image_kernel_regularizer,
                bias_regularizer=self.vector_to_image_bias_regularizer,
                use_equalized_learning_rate=(
                    recon_dict["use_equalized_learning_rate"]
                ),
                name="{}_base_layers_conv2d_{}_{}x{}_{}_{}".format(
                    self.vector_to_image_name,
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
                alpha=self.params["generator"]["leaky_relu_alpha"],
                name="{}_base_layers_leaky_relu_{}".format(
                    self.vector_to_image_name, i
                )
            )
            for i in range(len(conv_block))
        ]

        return base_conv_layers, base_leaky_relu_layers

    def _create_vector_to_image_growth_conv_layer_block(self, block_idx):
        """Creates vector_to_image growth conv layer block.

        Args:
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block's conv layers and list of growth block's
                leaky relu layers.
        """
        recon_dict = self.params["training"]["reconstruction"]
        # Get conv block layer properties.
        conv_block = self.params["generator"]["growth_conv_blocks"][block_idx]

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
                    if recon_dict["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.vector_to_image_kernel_regularizer,
                bias_regularizer=self.vector_to_image_bias_regularizer,
                use_equalized_learning_rate=(
                    recon_dict["use_equalized_learning_rate"]
                ),
                name="{}_growth_layers_conv2d_{}_{}_{}x{}_{}_{}".format(
                    self.vector_to_image_name,
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
                alpha=self.params["generator"]["leaky_relu_alpha"],
                name="{}_growth_layers_leaky_relu_{}_{}".format(
                    self.vector_to_image_name, block_idx, i
                )
            )
            for i in range(len(conv_block))
        ]

        return growth_conv_layers, growth_leaky_relu_layers

    def _create_vector_to_image_to_rgb_layers(self):
        """Creates vector_to_image toRGB layers of 1x1 convs.

        Returns:
            List of toRGB 1x1 conv layers.
        """
        recon_dict = self.params["training"]["reconstruction"]
        # Dictionary containing possible final activations.
        final_activation_set = {"sigmoid", "relu", "tanh"}

        # Get toRGB layer properties.
        to_rgb = [
            self.params["generator"]["to_rgb_layers"][i][0][:]
            for i in range(
                len(self.params["generator"]["to_rgb_layers"])
            )
        ]

        # Create list to hold toRGB 1x1 convs.
        to_rgb_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=to_rgb[i][3],
                kernel_size=to_rgb[i][0:2],
                strides=to_rgb[i][4:6],
                padding="same",
                activation=(
                    self.params["generator"]["final_activation"].lower()
                    if self.params["generator"]["final_activation"].lower()
                    in final_activation_set
                    else None
                ),
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if recon_dict["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.vector_to_image_kernel_regularizer,
                bias_regularizer=self.vector_to_image_bias_regularizer,
                use_equalized_learning_rate=(
                    recon_dict["use_equalized_learning_rate"]
                ),
                name="{}_to_rgb_layers_conv2d_{}_{}x{}_{}_{}".format(
                    self.vector_to_image_name,
                    i,
                    to_rgb[i][0],
                    to_rgb[i][1],
                    to_rgb[i][2],
                    to_rgb[i][3]
                )
            )
            for i in range(len(to_rgb))
        ]

        return to_rgb_conv_layers

    def _create_vector_to_image_layers(self):
        """Creates vector_to_image layers.
        """
        self.vector_to_image_input_layer = (
            self._create_vector_to_image_input_layer()
        )

        self.projection_dense_layer = (
            self._create_vector_to_image_projection_dense_layer()
        )

        (base_conv_layers,
         base_leaky_relu_layers) = (
            self._create_vector_to_image_base_conv_layer_block()
        )
        self.vector_to_image_conv_layers.append(base_conv_layers)
        self.vector_to_image_leaky_relu_layers.append(base_leaky_relu_layers)

        for block_idx in range(
            len(self.params["generator"]["growth_conv_blocks"])
        ):
            (growth_conv_layers,
             growth_leaky_relu_layers
             ) = self._create_vector_to_image_growth_conv_layer_block(
                block_idx
            )

            self.vector_to_image_conv_layers.append(growth_conv_layers)
            self.vector_to_image_leaky_relu_layers.append(
                growth_leaky_relu_layers
            )

        self.vector_to_image_weighted_sum_layer = custom_layers.WeightedSum(
            alpha=self.alpha_var,
            name="weighted_sum_{}".format(self.vector_to_image_name)
        )

        self.to_rgb_conv_layers = self._create_vector_to_image_to_rgb_layers()

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _maybe_use_pixel_norm(self, inputs, epsilon=1e-8):
        """Decides based on user parameter whether to use pixel norm or not.

        Args:
            epsilon: float, small value to add to denominator for numerical
                stability.
            inputs: tensor, image to potentially be normalized.

        Returns:
            Pixel normalized feature vectors if using pixel norm, else
                original feature vectors.
        """
        if self.params["generator"]["use_pixel_norm"]:
            return custom_layers.PixelNormalization(epsilon=epsilon)(
                inputs=inputs
            )
        return inputs

    def _vector_to_image_project_latent_vectors(self, latent_vectors):
        """Projects latent vectors into images.

        Args:
            latent_vectors: tensor, latent vector inputs of shape
                (batch_size, latent_size).

        Returns:
            Projected image of latent vector inputs.
        """
        # Possibly normalize latent vectors.
        if self.params["generator"]["normalize_latents"]:
            latent_vectors = self._maybe_use_pixel_norm(
                inputs=latent_vectors,
                epsilon=self.params["generator"]["pixel_norm_epsilon"]
            )

        projection_height = self.params["generator"]["projection_dims"][0]
        projection_width = self.params["generator"]["projection_dims"][1]
        projection_depth = self.params["generator"]["projection_dims"][2]

        # shape = (
        #     batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection = self.projection_dense_layer(inputs=latent_vectors)

        projection_leaky_relu = tf.keras.layers.LeakyReLU(
            alpha=self.params["generator"]["leaky_relu_alpha"],
            name="projection_leaky_relu"
        )(inputs=projection)

        # Reshape projection into "image".
        # shape = (
        #     batch_size,
        #     projection_height,
        #     projection_width,
        #     projection_depth
        # )
        projected_image = tf.reshape(
            tensor=projection_leaky_relu,
            shape=(
                -1, projection_height, projection_width, projection_depth
            ),
            name="projected_image"
        )

        # Possibly add pixel normalization to image.
        projected_image = self._maybe_use_pixel_norm(
            inputs=projected_image,
            epsilon=self.params["generator"]["pixel_norm_epsilon"]
        )

        return projected_image

    def _upsample_image(self, image, orig_img_size, block_idx):
        """Upsamples intermediate image.

        Args:
            image: tensor, image created by vector_to_image conv block.
            orig_img_size: list, the height and width dimensions of the
                original image before any growth.
            block_idx: int, index of the current vector_to_image growth block.

        Returns:
            Upsampled image tensor.
        """
        # Upsample from s X s to 2s X 2s image.
        upsampled_image = tf.image.resize(
            images=image,
            size=tf.convert_to_tensor(
                value=orig_img_size,
                dtype=tf.int32
            ) * 2 ** block_idx,
            method="nearest",
            name="{}_growth_upsampled_image_{}_{}x{}_{}x{}".format(
                self.vector_to_image_name,
                block_idx,
                orig_img_size[0] * 2 ** (block_idx - 1),
                orig_img_size[1] * 2 ** (block_idx - 1),
                orig_img_size[0] * 2 ** block_idx,
                orig_img_size[1] * 2 ** block_idx
            )
        )

        return upsampled_image

    def _fused_conv2d_act_pixel_norm_block(
        self, conv_layer, activation_layer, inputs
    ):
        """Fused Conv2D, activation, and pixel norm operation block.

        Args:
            conv_layer: instance of `Conv2D` layer.
            activation_layer: instance of `Layer`, such as LeakyRelu layer.
            inputs: tensor, inputs to fused block.

        Returns:
            Output tensor of fused block.
        """
        # Perform convolution with no activation.
        network = conv_layer(inputs=inputs)

        # Now apply activation to convolved inputs.
        network = activation_layer(inputs=network)

        # Possibly add pixel normalization to image.
        network = self._maybe_use_pixel_norm(
            inputs=network,
            epsilon=self.params["generator"]["pixel_norm_epsilon"]
        )

        return network

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_vector_to_image_projection_and_base_block(self, inputs):
        """Creates vector_to_image projection and base block.

        Returns:
            Output tensor of vector_to_image base block of shape
                (gpd[0], gpd[1], gpd[2]) where gpd is
                the generator_projection_dims parameter.
        """
        # Create the input layer to vector_to_image.
        # shape = (batch_size, latent_size)
        if not self.params["training"]["subclass_models"]:
            inputs = self.vector_to_image_input_layer

        # Project latent vectors.
        network = self._vector_to_image_project_latent_vectors(
            latent_vectors=inputs
        )

        # Get base block layers.
        base_conv_layers = self.vector_to_image_conv_layers[0]
        base_leaky_relu_layers = self.vector_to_image_leaky_relu_layers[0]

        if (
            self.params["generator"]["architecture"] == "GANomaly" and
            self.params["generator"]["GANomaly"]["use_unet_skip_connections"][0]
        ):
            unet_encoder_activation = self.unet_encoder_activations[0]
            network = tf.concat(
                values=[network, unet_encoder_activation], axis=-1
            )

        # Pass inputs through layer chain.
        for i in range(len(base_conv_layers)):
            network = self._fused_conv2d_act_pixel_norm_block(
                conv_layer=base_conv_layers[i],
                activation_layer=base_leaky_relu_layers[i],
                inputs=network
            )

        return network

    def _create_vector_to_image_growth_transition_weighted_sum(
        self, inputs, block_idx
    ):
        """Creates growth transition vec_to_img weighted_sum.

        Args:
            inputs: tensor, rank 4 tensor input to weighted sum block.
            block_idx: int, current block index of model progression.

        Returns:
            Rank 4 tensor of weighted sum between shrinking and growing block
                paths.
        """
        # Upsample most recent block conv image for both side chains.
        upsampled_block_conv = self._upsample_image(
            image=inputs,
            orig_img_size=self.params["generator"]["projection_dims"][0:2],
            block_idx=block_idx
        )

        # Growing side chain.
        growing_conv_layers = self.vector_to_image_conv_layers[block_idx]
        growing_leaky_relu_layers = (
            self.vector_to_image_leaky_relu_layers[block_idx]
        )
        growing_to_rgb_conv_layer = self.to_rgb_conv_layers[block_idx]

        # Pass inputs through layer chain.
        network = upsampled_block_conv

        if (
            self.params["generator"]["architecture"] == "GANomaly" and
            self.params["generator"]["GANomaly"]["use_unet_skip_connections"][block_idx]
        ):
            unet_encoder_activation = self.unet_encoder_activations[block_idx]
            network = tf.concat(
                values=[network, unet_encoder_activation], axis=-1
            )

        for i in range(0, len(growing_conv_layers)):
            network = self._fused_conv2d_act_pixel_norm_block(
                conv_layer=growing_conv_layers[i],
                activation_layer=growing_leaky_relu_layers[i],
                inputs=network
            )

        growing_to_rgb_conv = growing_to_rgb_conv_layer(inputs=network)

        # Shrinking side chain.
        shrinking_to_rgb_conv_layer = self.to_rgb_conv_layers[block_idx - 1]

        # Pass inputs through layer chain.
        shrinking_to_rgb_conv = shrinking_to_rgb_conv_layer(
            inputs=upsampled_block_conv
        )

        # Weighted sum.
        weighted_sum = self.vector_to_image_weighted_sum_layer(
            inputs=[growing_to_rgb_conv, shrinking_to_rgb_conv]
        )

        return weighted_sum

    def _create_vector_to_image_perm_growth_block_network(
        self, inputs, block_idx, stable
    ):
        """Creates vector_to_image permanent block network.

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
            self.vector_to_image_conv_layers[0:block_idx + stable]
        )
        permanent_leaky_relu_layers = (
            self.vector_to_image_leaky_relu_layers[0:block_idx + stable]
        )

        # Pass inputs through layer chain.
        network = inputs

        # Growth blocks require first prev conv layer's image upsampled.
        for i in range(1, len(permanent_conv_layers)):
            # Upsample previous block's image.
            network = self._upsample_image(
                image=network,
                orig_img_size=gen_params["projection_dims"][0:2],
                block_idx=i
            )

            if (
                gen_params["architecture"] == "GANomaly" and
                gen_params["GANomaly"]["use_unet_skip_connections"][i]
            ):
                unet_encoder_activation = self.unet_encoder_activations[i]
                network = tf.concat(
                    values=[network, unet_encoder_activation], axis=-1
                )

            block_conv_layers = permanent_conv_layers[i]
            block_leaky_relu_layers = permanent_leaky_relu_layers[i]
            for j in range(len(block_conv_layers)):
                network = self._fused_conv2d_act_pixel_norm_block(
                    conv_layer=block_conv_layers[j],
                    activation_layer=block_leaky_relu_layers[j],
                    inputs=network
                )

        return network

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _get_vector_to_image_base_model_outputs(
        self, inputs, training, block_idx
    ):
        """Builds vector_to_image base model.

        Args:
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        network = self._create_vector_to_image_projection_and_base_block(
            inputs
        )

        base_to_rgb_conv_layer = self.to_rgb_conv_layers[0]
        fake_images = base_to_rgb_conv_layer(inputs=network)

        return fake_images

    def _get_vector_to_image_growth_transition_model_outputs(
        self, inputs, training, block_idx
    ):
        """Builds vector_to_image growth transition model.

        Args:
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Base block doesn't need any upsampling so handle differently.
        network = self._create_vector_to_image_projection_and_base_block(inputs)

        # Permanent blocks.
        network = self._create_vector_to_image_perm_growth_block_network(
            inputs=network, block_idx=block_idx, stable=0
        )

        fake_images = (
            self._create_vector_to_image_growth_transition_weighted_sum(
                inputs=network, block_idx=block_idx
            )
        )

        return fake_images

    def _get_vector_to_image_growth_stable_model_outputs(self, inputs, training, block_idx):
        """Builds vector_to_image growth stable model.

        Args:
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Base block doesn't need any upsampling so handle differently.
        network = self._create_vector_to_image_projection_and_base_block(inputs)

        # Permanent blocks.
        network = self._create_vector_to_image_perm_growth_block_network(
            inputs=network, block_idx=block_idx, stable=1
        )

        # Get toRGB layer.
        to_rgb_conv_layer = self.to_rgb_conv_layers[block_idx]

        fake_images = to_rgb_conv_layer(inputs=network)

        return fake_images
