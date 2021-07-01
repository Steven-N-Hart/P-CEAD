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


class WeightScaledDense(tf.keras.layers.Dense):
    """Subclassing `Dense` layer to allow equalized learning rate scaling.

    Attributes:
        use_equalized_learning_rate: bool, if want to scale layer weights to
            equalize learning rate each forward pass.
    """
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        use_equalized_learning_rate=False,
        name=None,
        **kwargs
    ):
        """Initializes `Dense` layer.
        Args:
            units: Integer or Long, dimensionality of the output space.
            activation: Activation function (callable). Set it to None to
                maintain a linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            kernel_initializer: Initializer function for the weight matrix.
                If `None` (default), weights are initialized using the default
                initializer used by `tf.compat.v1.get_variable`.
            bias_initializer: Initializer function for the bias.
            kernel_regularizer: Regularizer function for the weight matrix.
            bias_regularizer: Regularizer function for the bias.
            activity_regularizer: Regularizer function for the output.
            kernel_constraint: An optional projection function to be applied to
                the kernel after being updated by an `Optimizer` (e.g. used to
                implement norm constraints or value constraints for layer
                weights). The function must take as input the unprojected
                variable and must return the projected variable (which must
                have the same shape). Constraints are not safe to use when
                doing asynchronous distributed training.
            bias_constraint: An optional projection function to be applied to
                the bias after being updated by an `Optimizer`.
            trainable: Boolean, if `True` also add variables to the graph
                collection `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
            use_equalized_learning_rate: bool, if want to scale layer weights
                to equalize learning rate each forward pass.
            name: String, the name of the layer. Layers with the same name will
                share weights, but to avoid mistakes we require reuse=True in
                such cases.
        """
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

        # Whether we will scale weights using He init every forward pass.
        self.use_equalized_learning_rate = use_equalized_learning_rate

    def call(self, inputs):
        """Calls layer and returns outputs.

        Args:
            inputs: tensor, input tensor of shape (batch_size, features).
        """
        if self.use_equalized_learning_rate:
            # Scale kernel weights by He init fade-in constant.
            kernel_shape = tuple(x for x in self.kernel.shape)
            fan_in = kernel_shape[0]
            he_constant = tf.sqrt(x=2. / float(fan_in))
            kernel = self.kernel * he_constant
        else:
            kernel = self.kernel

        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(
                a=inputs, b=kernel, axes=[[rank - 1], [0]]
            )
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(shape=output_shape)
        else:
            inputs = tf.cast(x=inputs, dtype=self._compute_dtype)
            if isinstance(inputs, tf.SparseTensor):
                outputs = tf.sparse_tensor_dense_matmul(sp_a=inputs, b=kernel)
            else:
                outputs = tf.matmul(a=inputs, b=kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(value=outputs, bias=self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable

        return outputs


class WeightScaledConv2D(tf.keras.layers.Conv2D):
    """Subclassing `Conv2D` layer to allow equalized learning rate scaling.

    Attributes:
        use_equalized_learning_rate: bool, if want to scale layer weights to
            equalize learning rate each forward pass.
    """
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format="channels_last",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        use_equalized_learning_rate=False,
        name=None,
        **kwargs
    ):
        """Initializes `Conv2D` layer.
        Args:
            filters: Integer, the dimensionality of the output space (i.e. the
                number of filters in the convolution).
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                height and width of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the height and
                width.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with
                specifying any `dilation_rate` value != 1.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string, one of `channels_last` (default) or
                `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, height, width, channels)` while `channels_first`
                corresponds to inputs with shape
                `(batch, channels, height, width)`.
            dilation_rate: An integer or tuple/list of 2 integers, specifying
                the dilation rate to use for dilated convolution.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Currently, specifying any `dilation_rate` value != 1 is
                incompatible with specifying any stride value != 1.
            activation: Activation function. Set it to None to maintain a
                linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            kernel_initializer: An initializer for the convolution kernel.
            bias_initializer: An initializer for the bias vector. If None, the
                default initializer will be used.
            kernel_regularizer: Optional regularizer for the convolution
                kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer function for the output.
            kernel_constraint: Optional projection function to be applied to
                the kernel after being updated by an `Optimizer` (e.g. used to
                implement norm constraints or value constraints for layer
                weights). The function must take as input the unprojected
                variable and must return the projected variable (which must
                have the same shape). Constraints are not safe to use when
                doing asynchronous distributed training.
            bias_constraint: Optional projection function to be applied to the
                bias after being updated by an `Optimizer`.
            trainable: Boolean, if `True` also add variables to the graph
                collection `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
            use_equalized_learning_rate: bool, if want to scale layer weights
                to equalize learning rate each forward pass.
            name: A string, the name of the layer.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

        # Whether we will scale weights using He init every forward pass.
        self.use_equalized_learning_rate = use_equalized_learning_rate

    def call(self, inputs):
        """Calls layer and returns outputs.
        Args:
            inputs: tensor, input tensor of shape
                [batch_size, height, width, channels].
        """
        if self.use_equalized_learning_rate:
            # Scale kernel weights by He init constant.
            kernel_shape = tuple(x for x in self.kernel.shape)
            fan_in = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
            he_constant = tf.sqrt(x=2. / float(fan_in))
            kernel = self.kernel * he_constant
        else:
            kernel = self.kernel

        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == "channels_first":
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = tf.reshape(
                        tensor=self.bias, shape=(1, self.filters, 1)
                    )
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(
                        value=outputs, bias=self.bias, data_format="NCHW"
                    )
            else:
                outputs = tf.nn.bias_add(
                    value=outputs, bias=self.bias, data_format="NHWC"
                )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class PixelNormalization(tf.keras.layers.Layer):
    """Normalizes the feature vector in each pixel to unit length.

    Attributes:
        epsilon: float, small value to add to denominator for numerical
            stability.
    """
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        """Calls PixelNormalization layer with given inputs tensor.

        Args:
            inputs: tensor, image feature vectors.

        Returns:
            Pixel normalized feature vector tensor.
        """
        return inputs * tf.math.rsqrt(
            x=tf.add(
                x=tf.reduce_mean(
                    input_tensor=tf.square(x=inputs), axis=-1, keepdims=True
                ),
                y=self.epsilon
            )
        )


class WeightedSum(tf.keras.layers.Layer):
    """Weighted sum between two tensors of equal shape.

    Attributes:
        alpha: tf.Variable, used in growth transition network's weighted sum,
            linearly scales through range [0., 1.].
    """
    def __init__(self, alpha, name):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, inputs):
        """Calls WeightedSum layer with given 2-tuple of input tensors.

        Args:
            inputs: 2-tuple, input tensors to weighted sum.

        Returns:
            Weighted sum tensor of both inputs.
        """
        return inputs[0] * self.alpha + inputs[1] * (1.0 - self.alpha)


class MiniBatchStdDev(tf.keras.layers.Layer):
    """Adds minibatch's stddev as a additional filter to images.

    Attributes:
        params: dict, user passed parameters.
        group_size: int, size of image groups.
        batch_size: tf.int64 tensor, the dynamic batch size (in case
            of partial batch).
        tile_multiples: list, length 4, used to tile input to final shape
            input_dims[i] * mutliples[i].
    """
    def __init__(self, params, name):
        super().__init__(name=name)
        self.params = params

        # Get group size.
        self.group_size = (
            self.params["group_size"] if self.params["group_size"] > 0 else 4
        )

        self.batch_size = None
        self.tile_multiples = []

    def _minibatch_stddev_common(self, variance):
        """Adds minibatch stddev feature map to image using grouping.

        This is the code that is common between the grouped and ungroup
        minibatch stddev functions.

        Args:
            variance: tensor, variance of minibatch or minibatch groups.

        Returns:
            Minibatch standard deviation feature map image added to
                channels of shape
                [batch_size, image_height, image_width, 1].
        """
        # Calculate standard deviation over the group plus small epsilon.
        # shape = (
        #     {"grouped": batch_size / group_size, "ungrouped": 1},
        #     image_size,
        #     image_size,
        #     num_channels
        # )
        stddev = tf.sqrt(x=variance + 1e-8, name="minibatch_stddev")

        # Take average over feature maps and pixels.
        if self.params["use_averaging"]:
            # grouped shape = (batch_size / group_size, 1, 1, 1)
            # ungrouped shape = (1, 1, 1, 1)
            stddev = tf.reduce_mean(
                input_tensor=stddev,
                axis=[1, 2, 3],
                keepdims=True,
                name="minibatch_stddev_average"
            )

        # Replicate over group and pixels.
        # shape = (batch_size, image_size, image_size, 1)
        stddev_feature_map = tf.tile(
            input=stddev,
            multiples=self.tile_multiples,
            name="minibatch_stddev_feature_map"
        )

        return stddev_feature_map

    def _grouped_minibatch_stddev(self, inputs):
        """Adds minibatch stddev feature map to image using grouping.

        Args:
            inputs: tf.float32 tensor, image of shape
                [batch_size, image_height, image_width, num_channels].

        Returns:
            Minibatch standard deviation feature map image added to
                channels of shape
                [batch_size, image_height, image_width, 1].
        """
        # The group size should be less than or equal to the batch size.
        group_size = tf.minimum(x=self.group_size, y=self.batch_size)

        # Split minibatch into M groups of size group_size, rank 5 tensor.
        # shape = (
        #     group_size,
        #     batch_size / group_size,
        #     image_size,
        #     image_size,
        #     num_channels
        # )
        grouped_image = tf.reshape(
            tensor=inputs,
            shape=tuple([group_size, -1] + list(inputs.shape[1:])),
            name="grouped_image"
        )

        # Find the mean of each group.
        # shape = (
        #     1,
        #     batch_size / group_size,
        #     image_size,
        #     image_size,
        #     num_channels
        # )
        grouped_mean = tf.reduce_mean(
            input_tensor=grouped_image,
            axis=0,
            keepdims=True,
            name="grouped_mean"
        )

        # Center each group using the mean.
        # shape = (
        #     group_size,
        #     batch_size / group_size,
        #     image_size,
        #     image_size,
        #     num_channels
        # )
        centered_grouped_image = tf.subtract(
            x=grouped_image, y=grouped_mean, name="centered_grouped_image"
        )

        # Calculate variance over group.
        # shape = (
        #     batch_size / group_size, image_size, image_size, num_channels
        # )
        grouped_variance = tf.reduce_mean(
            input_tensor=tf.square(x=centered_grouped_image),
            axis=0,
            name="grouped_variance"
        )

        # Get stddev image using ops common to both grouped & ungrouped.
        self.tile_multiples = [group_size] + list(inputs.shape[1:3]) + [1]
        stddev_feature_map = self._minibatch_stddev_common(
            variance=grouped_variance
        )

        return stddev_feature_map

    def _ungrouped_minibatch_stddev(self, inputs):
        """Adds minibatch stddev feature map added to image channels.

        Args:
            inputs: tensor, image of shape
                [batch_size, image_height, image_width, num_channels].

        Returns:
            Minibatch standard deviation feature map image added to
                channels of shape
                [batch_size, image_height, image_width, 1].
        """
        # Find the mean of each group.
        # shape = (1, image_size, image_size, num_channels)
        mean = tf.reduce_mean(
            input_tensor=inputs, axis=0, keepdims=True, name="mean"
        )

        # Center each group using the mean.
        # shape = (batch_size, image_size, image_size, num_channels)
        centered_image = tf.subtract(
            x=inputs, y=mean, name="centered_image"
        )

        # Calculate variance over group.
        # shape = (1, image_size, image_size, num_channels)
        variance = tf.reduce_mean(
            input_tensor=tf.square(x=centered_image),
            axis=0,
            keepdims=True,
            name="variance"
        )

        # Get stddev image using ops common to both grouped & ungrouped.
        self.tile_multiples = [self.batch_size] + list(inputs.shape[1:3]) + [1]
        stddev_feature_map = self._minibatch_stddev_common(variance=variance)

        return stddev_feature_map

    @tf.function()
    def _minibatch_stddev(self, inputs):
        """Adds minibatch stddev feature map added to image.

        Args:
            inputs: tensor, image of shape
                [batch_size, image_height, image_width, num_channels].

        Returns:
            Image with minibatch standard deviation feature map added to
                channels of shape
                [batch_size, image_height, image_width, num_channels + 1].
        """
        # Get batch size.
        self.batch_size = tf.shape(inputs)[0]

        if (self.batch_size % self.group_size == 0 or
            self.batch_size < self.group_size
           ):
            stddev_feature_map = self._grouped_minibatch_stddev(
                inputs=inputs
            )
        else:
            stddev_feature_map = self._ungrouped_minibatch_stddev(
                inputs=inputs
            )

        # Append new feature map to image.
        # shape = (batch_size, image_height, image_width, num_channels + 1)
        appended_image = tf.concat(
            values=[inputs, stddev_feature_map],
            axis=-1,
            name="appended_image"
        )

        return appended_image

    def call(self, inputs):
        """Calls minibatch stddev layer with inputs and training flag.

        Args:
            inputs: tensor, image of shape
                [batch_size, image_height, image_width, num_channels].

        Returns:
            Image possibly with minibatch standard deviation feature map added
                to channels of shape
                [batch_size, image_height, image_width, num_channels + 1].
        """
        if self.params["use_minibatch_stddev"]:
            return self._minibatch_stddev(inputs)

        return inputs
