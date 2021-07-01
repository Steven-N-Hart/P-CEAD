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


class ExportBerg(object):
    """Class used for exporting Berg model objects.
    """
    def __init__(self):
        """Instantiate instance of `ExportBerg`.
        """
        pass

    def export_using_z_input_berg_encoded_generated_images(
        self,
        growth_idx,
        export_outputs,
        generator_model,
        encoded_generated_logits
    ):
        """Adds encoded generated images to export for berg Z inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoded_generated_logits: tensor, E(G(z)) of shape
                (batch_size, latent_size).
        """
        encoded_generated_images = tf.identity(
            input=generator_model(
                inputs=encoded_generated_logits, training=False
            ),
            name="encoded_generated_images_{}".format(
                growth_idx
            )
        )

        # shape = (batch_size, height, width, depth).
        # range = [-1., 1.].
        export_outputs.append(encoded_generated_images)

    def export_using_z_input_berg_encoded_generated_logits_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        encoder_model,
        generated_images
    ):
        """Adds encoded generated logits & beyond to export for berg Z inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
            generated_images: tensor, G(z) of shape
                (batch_size, height, width, depth).
        """
        encoded_generated_logits = tf.identity(
            input=encoder_model(inputs=generated_images, training=False),
            name="encoded_generated_logits_{}".format(growth_idx)
        )

        if export_dict["export_encoded_generated_logits"]:
            # shape = (batch_size, 1).
            # range = (-inf, inf).
            export_outputs.append(encoded_generated_logits)

        if export_dict["export_encoded_generated_images"]:
            self.export_using_z_input_berg_encoded_generated_images(
                growth_idx,
                export_outputs,
                generator_model,
                encoded_generated_logits
            )

    def export_using_z_input_berg_generated_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        encoder_model,
        Z
    ):
        """Adds generated images & beyond to export for berg Z inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
            Z: tensor, latent inputs of shape (batch_size, latent_size).
        """
        generated_images = tf.identity(
            input=generator_model(inputs=Z, training=False),
            name="generated_images_{}".format(growth_idx)
        )

        if export_dict["export_generated_images"]:
            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(generated_images)

        if self.params["encoder"]["create"] and any(
            [
                export_dict["export_encoded_generated_logits"],
                export_dict["export_encoded_generated_images"]
            ]
        ):
            self.export_using_z_input_berg_encoded_generated_logits_and_beyond(
            growth_idx,
            export_dict,
            export_outputs,
            generator_model,
            encoder_model,
            generated_images
        )

    def export_using_z_input_berg_z_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_inputs_Z,
        export_outputs,
        generator_model,
        encoder_model
    ):
        """Adds Z & beyond to export for berg Z inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_inputs_Z: list, stores Z export inputs.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
        """
        if self.params["training"]["subclass_models"]:
            Z = generator_model.vector_to_image_input_layer
        else:
            Z = generator_model.inputs[0]
        export_inputs_Z.append(Z)

        if export_dict["export_Z"]:
            # shape = (batch_size, generator_latent_size).
            # range = (-inf, inf).
            export_outputs.append(
                tf.identity(input=Z, name="Z_{}".format(growth_idx))
            )

        if any(
            [
                export_dict["export_generated_images"],
                (
                    self.params["encoder"]["create"] and
                    any(
                        [
                            export_dict["export_encoded_generated_logits"],
                            export_dict["export_encoded_generated_images"]
                        ]
                    )
                )
            ]
        ):
            self.export_using_z_input_berg_generated_images_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                generator_model,
                encoder_model,
                Z
            )

    def export_using_z_input_berg(
        self,
        growth_idx,
        export_dict,
        export_inputs_Z,
        export_outputs,
        generator_model,
        encoder_model
    ):
        """Adds inputs and outputs to export for berg Z inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_inputs_Z: list, stores Z export inputs.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
        """
        if any(
            [
                export_dict["export_Z"],
                export_dict["export_generated_images"],
                (
                    self.params["encoder"]["create"] and
                    any(
                        [
                            export_dict["export_encoded_generated_logits"],
                            export_dict["export_encoded_generated_images"]
                        ]
                    )
                )
            ]
        ):
            self.export_using_z_input_berg_z_and_beyond(
                growth_idx,
                export_dict,
                export_inputs_Z,
                export_outputs,
                generator_model,
                encoder_model
            )

    def export_using_query_images_input_berg_anomaly_images(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        query_images,
        encoded_images
    ):
        """Adds anomaly image outputs to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            encoded_images: tensor, G(E(x)) of shape
                (batch_size, height, width, depth).
        """
        if export_dict["export_query_anomaly_images_sigmoid"]:
            anomaly_images_sigmoid = self.anomaly_localization_sigmoid(
                query_images=query_images,
                encoded_images=encoded_images
            )

            anomaly_images_sigmoid = tf.identity(
                input=anomaly_images_sigmoid,
                name="query_anomaly_images_sigmoid_{}".format(growth_idx)
            )

            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(anomaly_images_sigmoid)

        if export_dict["export_query_anomaly_images_linear"]:
            anomaly_images_linear = self.anomaly_localization_linear(
                query_images=query_images,
                encoded_images=encoded_images
            )

            anomaly_images_linear = tf.identity(
                input=anomaly_images_linear,
                name="query_anomaly_images_linear_{}".format(growth_idx)
            )

            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(anomaly_images_linear)

    def export_using_query_images_input_berg_mahalanobis_distance_images(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        mahalanobis_distances
    ):
        """Adds Mahalanobis distance image outputs to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            mahalanobis_distances: tensor, Mahalanobis distance of image
                errors of shape (batch_size, height, width).
        """
        if export_dict["export_query_mahalanobis_distance_images_sigmoid"]:
            sigmoid_mahalanobis_distances_sigmoid = tf.math.sigmoid(
                x=mahalanobis_distances
            )

            mahalanobis_distance_images_sigmoid = (
                self.zero_center_sigmoid_absolute_values(
                    absolute_values=sigmoid_mahalanobis_distances_sigmoid
                )
            )

            mahalanobis_distance_images_sigmoid = tf.expand_dims(
                input=mahalanobis_distance_images_sigmoid,
                axis=-1,
                name="query_mahalanobis_distance_images_sigmoid_{}".format(
                    growth_idx
                )
            )

            # shape = (batch_size, height, width, 1).
            # range = [-1., 1.].
            export_outputs.append(mahalanobis_distance_images_sigmoid)

        if export_dict["export_query_mahalanobis_distance_images_linear"]:
            # Min-max normalize scores to scale range to [0, 1].
            normalized_distances = self.minmax_normalization(
                X=tf.expand_dims(input=mahalanobis_distances, axis=-1)
            )

            # Scale images to [-1, 1).
            mahalanobis_distance_images_linear = (
                normalized_distances * 2. - 1.
            )

            mahalanobis_distance_images_linear = tf.identity(
                input=mahalanobis_distance_images_linear,
                name="query_mahalanobis_distance_images_linear_{}".format(
                    growth_idx
                )
            )

            # shape = (batch_size, height, width, 1).
            # range = [-1., 1.].
            export_outputs.append(mahalanobis_distance_images_linear)

    def export_using_query_images_input_berg_pixel_anomaly_flag_counts_and_percentages(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        pixel_anomaly_flag_images
    ):
        """Adds pixel anomaly flag counts & percentages to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            pixel_anomaly_flag_images: tensor, pixel anomaly flags image of
                shape (batch_size, height, width, 1).
        """
        counts = tf.math.reduce_sum(
            input_tensor=tf.cast(
                x=pixel_anomaly_flag_images == 1,
                dtype=tf.int32
            ),
            axis=(1, 2, 3),
            name="query_pixel_anomaly_flag_counts"
        )

        if export_dict["export_query_pixel_anomaly_flag_counts"]:
            # shape = (batch_size,).
            # range = [0, inf).
            export_outputs.append(counts)

        if export_dict["export_query_pixel_anomaly_flag_percentages"]:
            # shape = (batch_size,).
            # range = [0., inf).
            export_outputs.append(
                tf.identity(
                    input=tf.math.divide_no_nan(
                        x=tf.cast(x=counts, dtype=tf.float32),
                        y=tf.cast(
                            x=tf.math.reduce_prod(
                                input_tensor=(
                                    pixel_anomaly_flag_images.shape[1:]
                                )
                            ),
                            dtype=tf.float32
                        )
                    ),
                    name="query_pixel_anomaly_flag_percentiles"
                )
            )
            

    def export_using_query_images_input_berg_pixel_anomaly_flag_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        mahalanobis_distances
    ):
        """Adds pixel anomaly flag image outputs to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            mahalanobis_distances: tensor, Mahalanobis distance of image
                errors of shape (batch_size, height, width).
        """
        pixel_anomaly_flag_images = tf.expand_dims(
            input=tf.where(
                condition=tf.greater(
                    x=mahalanobis_distances, y=self.dynamic_threshold
                ),
                x=tf.ones_like(input=mahalanobis_distances, dtype=tf.float32),
                y=-tf.ones_like(input=mahalanobis_distances, dtype=tf.float32)
            ),
            axis=-1,
            name="query_pixel_anomaly_flag_images_{}".format(growth_idx)
        )

        # shape = (batch_size, height, width, 1).
        # range = [-1., 1.].
        export_outputs.append(pixel_anomaly_flag_images)

        if any(
            [
                export_dict["export_query_pixel_anomaly_flag_counts"],
                export_dict["export_query_pixel_anomaly_flag_percentages"]
            ]
        ):
            self.export_using_query_images_input_berg_pixel_anomaly_flag_counts_and_percentages(
                growth_idx,
                export_dict,
                export_outputs,
                pixel_anomaly_flag_images
            )

    def export_using_query_images_input_berg_mahalanobis_distances_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        query_images,
        encoded_images
    ):
        """Adds Mahalanobis distance outputs & beyond to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            encoded_images: tensor, G(E(x)) of shape
                (batch_size, height, width, depth).
        """
        errors = self.reshape_image_absolute_errors(
            errors=tf.abs(x=query_images - encoded_images)
        )

        mahalanobis_distances = self.batch_mahalanobis_distance(
            batch_matrix=errors,
            mu=self.network_objects["error_distribution"].col_means_vector
        )

        mahalanobis_distances_reshaped = tf.reshape(
            tensor=mahalanobis_distances,
            shape=(-1, query_images.shape[1], query_images.shape[2]),
            name="query_mahalanobis_distances_{}".format(growth_idx)
        )

        # shape = (batch_size, height, width).
        # range = (0., inf).
        export_outputs.append(mahalanobis_distances_reshaped)

        if any(
            [
                export_dict["export_query_mahalanobis_distance_images_sigmoid"],
                export_dict["export_query_mahalanobis_distance_images_linear"]
            ]
        ):
            self.export_using_query_images_input_berg_mahalanobis_distance_images(
                growth_idx,
                export_dict,
                export_outputs,
                mahalanobis_distances=mahalanobis_distances_reshaped
            )

        if (
            self.training_phase.numpy() > 1 and any(
                [
                    export_dict["export_query_pixel_anomaly_flag_images"],
                    export_dict["export_query_pixel_anomaly_flag_counts"],
                    export_dict["export_query_pixel_anomaly_flag_percentages"]
                ]
            )
        ):
            self.export_using_query_images_input_berg_pixel_anomaly_flag_images_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                mahalanobis_distances=mahalanobis_distances_reshaped
            )

    def export_using_query_images_input_berg_anomaly_scores_and_flags(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        query_images,
        encoded_logits,
        encoded_images
    ):
        """Adds anomaly score & flag outputs to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            encoded_logits: tensor, E(x) of shape (batch_size, latent_size).
            encoded_images: tensor, G(E(x)) of shape
                (batch_size, height, width, depth).
        """
        anomaly_scores, anomaly_flags = self.anomaly_detection(
            query_images=query_images,
            encoded_logits=encoded_logits,
            encoded_images=encoded_images
        )

        if export_dict["export_query_anomaly_scores"]:
            # shape = (batch_size,).
            # range = (-inf, inf).
            export_outputs.append(
                tf.identity(
                    input=anomaly_scores,
                    name="query_anomaly_scores_{}".format(
                        growth_idx
                    )
                )
            )

        if export_dict["export_query_anomaly_flags"]:
            # shape = (batch_size,).
            # range = [0, 1].
            export_outputs.append(
                tf.identity(
                    input=anomaly_flags,
                    name="query_anomaly_flags_{}".format(
                        growth_idx
                    )
                )
            )

    def export_using_query_images_input_berg_encoded_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        query_images,
        encoded_logits
    ):
        """Adds encoded image outputs & beyond to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            encoded_logits: tensor, E(x) of shape (batch_size, latent_size).
        """
        encoded_images = tf.identity(
            input=generator_model(inputs=encoded_logits, training=False),
            name="query_encoded_images_{}".format(growth_idx)
        )

        if export_dict["export_query_encoded_images"]:
            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(encoded_images)

        if any(
            [
                export_dict["export_query_anomaly_images_sigmoid"],
                export_dict["export_query_anomaly_images_linear"]
            ]
        ):
            self.export_using_query_images_input_berg_anomaly_images(
                growth_idx,
                export_dict,
                export_outputs,
                query_images,
                encoded_images
            )

        if any(
            [
                (
                    self.training_phase.numpy() > 0 and any(
                        [
                            export_dict["export_query_mahalanobis_distances"],
                            export_dict["export_query_mahalanobis_distance_images_sigmoid"],
                            export_dict["export_query_mahalanobis_distance_images_linear"]
                        ]
                    )
                ),
                (
                    self.training_phase.numpy() > 1 and any(
                        [
                            export_dict["export_query_pixel_anomaly_flag_images"],
                            export_dict["export_query_pixel_anomaly_flag_counts"],
                            export_dict["export_query_pixel_anomaly_flag_percentages"]
                        ]
                    )
                )
            ]
        ):
            self.export_using_query_images_input_berg_mahalanobis_distances_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                query_images,
                encoded_images
            )

        if any(
            [
                export_dict["export_query_anomaly_scores"],
                export_dict["export_query_anomaly_flags"]
            ]
        ):
            self.export_using_query_images_input_berg_anomaly_scores_and_flags(
                growth_idx,
                export_dict,
                export_outputs,
                query_images,
                encoded_logits,
                encoded_images
            )

    def export_using_query_images_input_berg_encoded_logits_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        encoder_model,
        query_images
    ):
        """Adds encoded logit outputs & beyond to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
        """
        encoded_logits = tf.identity(
            input=encoder_model(inputs=query_images, training=False),
            name="query_encoded_logits_{}".format(growth_idx)
        )

        if export_dict["export_query_encoded_logits"]:
            # shape = (batch_size, generator_latent_size).
            # range = (-inf, inf).
            export_outputs.append(encoded_logits)

        if any(
            [
                export_dict["export_query_encoded_images"],
                export_dict["export_query_anomaly_images_sigmoid"],
                export_dict["export_query_anomaly_images_linear"],
                (
                    self.training_phase.numpy() > 0 and any(
                        [
                            export_dict["export_query_mahalanobis_distances"],
                            export_dict["export_query_mahalanobis_distance_images_sigmoid"],
                            export_dict["export_query_mahalanobis_distance_images_linear"]
                        ]
                    )
                ),
                (
                    self.training_phase.numpy() > 1 and any(
                        [
                            export_dict["export_query_pixel_anomaly_flag_images"],
                            export_dict["export_query_pixel_anomaly_flag_counts"],
                            export_dict["export_query_pixel_anomaly_flag_percentages"]
                        ]
                    )
                ),
                export_dict["export_query_anomaly_scores"],
                export_dict["export_query_anomaly_flags"]
            ]
        ):
            self.export_using_query_images_input_berg_encoded_images_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                generator_model,
                query_images,
                encoded_logits
            )

    def export_using_query_images_input_berg_query_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_inputs_query_images,
        export_outputs,
        generator_model,
        encoder_model
    ):
        """Adds query image outputs & beyond to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_inputs_query_images: list, stores query image export
                inputs.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
        """
        # x.
        if self.params["training"]["subclass_models"]:
            block_idx = (growth_idx + 1) // 2
            query_images = (
                encoder_model.image_to_vector_input_layers[block_idx]
            )
        else:
            query_images = encoder_model.inputs[0]
        export_inputs_query_images.append(query_images)

        if export_dict["export_query_images"]:
            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(
                tf.identity(
                    input=query_images, name="query_images_{}".format(
                        growth_idx
                    )
                )
            )

        if self.params["encoder"]["create"] and any(
            [
                export_dict["export_query_encoded_logits"],
                export_dict["export_query_encoded_images"],
                export_dict["export_query_anomaly_images_sigmoid"],
                export_dict["export_query_anomaly_images_linear"],
                (
                    self.training_phase.numpy() > 0 and any(
                        [
                            export_dict["export_query_mahalanobis_distances"],
                            export_dict["export_query_mahalanobis_distance_images_sigmoid"],
                            export_dict["export_query_mahalanobis_distance_images_linear"]
                        ]
                    )
                ),
                (
                    self.training_phase.numpy() > 1 and any(
                        [
                            export_dict["export_query_pixel_anomaly_flag_images"],
                            export_dict["export_query_pixel_anomaly_flag_counts"],
                            export_dict["export_query_pixel_anomaly_flag_percentages"]
                        ]
                    )
                ),
                export_dict["export_query_anomaly_scores"],
                export_dict["export_query_anomaly_flags"]
            ]
        ):
            self.export_using_query_images_input_berg_encoded_logits_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                generator_model,
                encoder_model,
                query_images
            )

    def export_using_query_images_input_berg(
        self,
        growth_idx,
        export_dict,
        export_inputs_query_images,
        export_outputs,
        generator_model,
        encoder_model
    ):
        """Adds inputs and outputs to export for berg query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_inputs_query_images: list, stores query image export
                inputs.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
        """
        if any(
            [
                export_dict["export_query_images"],
                export_dict["export_query_encoded_logits"],
                export_dict["export_query_encoded_images"],
                export_dict["export_query_anomaly_images_sigmoid"],
                export_dict["export_query_anomaly_images_linear"],
                (
                    self.training_phase.numpy() > 0 and any(
                        [
                            export_dict["export_query_mahalanobis_distances"],
                            export_dict["export_query_mahalanobis_distance_images_sigmoid"],
                            export_dict["export_query_mahalanobis_distance_images_linear"]
                        ]
                    )
                ),
                (
                    self.training_phase.numpy() > 1 and any(
                        [
                            export_dict["export_query_pixel_anomaly_flag_images"],
                            export_dict["export_query_pixel_anomaly_flag_counts"],
                            export_dict["export_query_pixel_anomaly_flag_percentages"]
                        ]
                    )
                ),
                export_dict["export_query_anomaly_scores"],
                export_dict["export_query_anomaly_flags"]
            ]
        ):
            self.export_using_query_images_input_berg_query_images_and_beyond(
                growth_idx,
                export_dict,
                export_inputs_query_images,
                export_outputs,
                generator_model,
                encoder_model
            )

    def create_serving_model_berg(self, growth_idx):
        """Creates Keras `Model` for serving Berg paper.

        Args:
            growth_idx: int, index of growth phase to export.
        """
        export_dict = self.params["export"]
        export_inputs_Z = []
        export_inputs_query_images = []
        export_outputs = []

        generator_model = (
            self.network_objects["generator"].models[growth_idx]
        )

        encoder_model = None
        if self.params["encoder"]["create"]:
            encoder_model = (
                self.network_objects["encoder"].models[growth_idx]
            )

        self.export_using_z_input_berg(
            growth_idx,
            export_dict,
            export_inputs_Z,
            export_outputs,
            generator_model,
            encoder_model
        )

        if self.params["encoder"]["create"]:
            self.export_using_query_images_input_berg(
                growth_idx,
                export_dict,
                export_inputs_query_images,
                export_outputs,
                generator_model,
                encoder_model
            )

        if export_dict["export_all_growth_phases"]:
            if export_inputs_Z:
                self.serving_inputs_Z = export_inputs_Z

            if export_inputs_query_images:
                query_images_name = export_inputs_query_images[0].name
                name_set = self.serving_inputs_query_images_names_set
                if query_images_name not in name_set:
                    self.serving_inputs_query_images.extend(
                        export_inputs_query_images
                    )
                    self.serving_inputs_query_images_names_set.add(
                        query_images_name
                    )

            self.serving_outputs.extend(export_outputs)

        serving_model = tf.keras.Model(
            inputs=export_inputs_Z + export_inputs_query_images,
            outputs=export_outputs,
            name="serving_model_growth_{}_epoch_{}".format(
                growth_idx, self.epoch_idx
            )
        )

        if export_dict["print_serving_model_summaries"]:
            print("\nserving_model = {}".format(serving_model.summary()))

        return serving_model

    def create_serving_models_berg(self):
        """Creates Keras `Model`s for serving for Berg paper.

        Args:
            growth_idx: int, index of growth phase to export.
        """
        if self.params["export"]["export_all_growth_phases"]:
            # Reset.
            self.serving_inputs_query_images_names_set.clear()
            self.serving_inputs_query_images.clear()
            self.serving_outputs.clear()
            for i in range(self.num_growths):
                self.create_serving_model_berg(growth_idx=i)

            if self.params["export"]["print_serving_model_summaries"]:
                print(
                    "create_serving_models: serving_inputs_Z = {}".format(
                        self.serving_inputs_Z
                    )
                )
                print(
                    "create_serving_models: {} = {}".format(
                        "serving_inputs_query_images",
                        self.serving_inputs_query_images
                    )
                )
                print(
                    "create_serving_models: serving_outputs = {}".format(
                        self.serving_outputs
                    )
                )

            self.serving_model = tf.keras.Model(
                inputs=(
                    self.serving_inputs_Z + self.serving_inputs_query_images
                ),
                outputs=self.serving_outputs,
                name="serving_model_all_growth_{}_epoch_{}".format(
                    self.growth_idx, self.epoch_idx
                )
            )
        else:
            self.serving_model = self.create_serving_model_berg(
                growth_idx=self.growth_idx
            )
