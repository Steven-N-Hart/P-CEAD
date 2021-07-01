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


class ExportGanomaly(object):
    """Class used for exporting GANomaly model objects.
    """
    def __init__(self):
        """Instantiate instance of `ExportGanomaly`.
        """
        pass

    def export_using_query_images_input_ganomaly_enc_encoded_images(
        self,
        growth_idx,
        export_outputs,
        generator_model,
        gen_encoded_images,
        enc_encoded_logits
    ):
        """Adds encoder encoded image outputs & beyond to export for GANomaly query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            gen_encoded_images: tensor, G(x) of shape
                (batch_size, height, width, depth).
            enc_encoded_logits: tensor, E(G(x)) of shape (batch_size, latent_size).
        """
        _, enc_encoded_images = generator_model(
            inputs=gen_encoded_images, training=False
        )
        enc_encoded_images = tf.identity(
            input=enc_encoded_images,
            name="query_enc_encoded_images_{}".format(growth_idx)
        )

        # shape = (batch_size, height, width, depth).
        # range = [-1., 1.].
        export_outputs.append(enc_encoded_images)

    def export_using_query_images_input_ganomaly_enc_encoded_logits_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        encoder_model,
        gen_encoded_images
    ):
        """Adds encoder encoded logit outputs & beyond to export for GANomaly query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
            gen_encoded_images: tensor, G(x) of shape
                (batch_size, height, width, depth).
        """
        # E(G(x)).
        enc_encoded_logits = tf.identity(
            input=encoder_model(inputs=gen_encoded_images, training=False),
            name="query_enc_encoded_logits_{}".format(
                growth_idx
            )
        )

        if export_dict["export_query_enc_encoded_logits"]:
            # shape = (batch_size, 1).
            # range = (-inf, inf).
            export_outputs.append(enc_encoded_logits)

        # Gd(E(G(x))).
        if export_dict["export_query_enc_encoded_images"]:
            self.export_using_query_images_input_ganomaly_enc_encoded_images(
                growth_idx,
                export_outputs,
                generator_model,
                gen_encoded_images,
                enc_encoded_logits
            )

    def export_using_query_images_input_ganomaly_anomaly_images(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        query_images,
        gen_encoded_images
    ):
        """Adds anomaly image outputs to export for GANomaly query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            gen_encoded_images: tensor, G(x) of shape
                (batch_size, height, width, depth).
        """
        if export_dict["export_query_anomaly_images_sigmoid"]:
            anomaly_images_sigmoid = self.anomaly_localization_sigmoid(
                query_images=query_images,
                encoded_images=gen_encoded_images
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
                encoded_images=gen_encoded_images
            )

            anomaly_images_linear = tf.identity(
                input=anomaly_images_linear,
                name="query_anomaly_images_linear_{}".format(growth_idx)
            )

            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(anomaly_images_linear)

    def export_using_query_images_input_ganomaly_mahalanobis_distance_images(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        mahalanobis_distances
    ):
        """Adds Mahalanobis distance image outputs to export for GANomaly query image inputs.

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

    def export_using_query_images_input_ganomaly_pixel_anomaly_flag_counts_and_percentages(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        pixel_anomaly_flag_images
    ):
        """Adds pixel anomaly flag counts & percentages to export for GANomaly query image inputs.

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
            

    def export_using_query_images_input_ganomaly_pixel_anomaly_flag_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        mahalanobis_distances
    ):
        """Adds pixel anomaly flag image outputs to export for GANomaly query image inputs.

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
            self.export_using_query_images_input_ganomaly_pixel_anomaly_flag_counts_and_percentages(
                growth_idx,
                export_dict,
                export_outputs,
                pixel_anomaly_flag_images
            )

    def export_using_query_images_input_ganomaly_mahalanobis_distances_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        query_images,
        gen_encoded_images
    ):
        """Adds Mahalanobis distance outputs & beyond to export for GANomaly query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            gen_encoded_images: tensor, G(x) of shape
                (batch_size, height, width, depth).
        """
        errors = self.reshape_image_absolute_errors(
            errors=tf.abs(x=query_images - gen_encoded_images)
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
        # range = [0., inf).
        export_outputs.append(mahalanobis_distances_reshaped)

        if any(
            [
                export_dict["export_query_mahalanobis_distance_images_sigmoid"],
                export_dict["export_query_mahalanobis_distance_images_linear"]
            ]
        ):
            self.export_using_query_images_input_ganomaly_mahalanobis_distance_images(
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
            self.export_using_query_images_input_ganomaly_pixel_anomaly_flag_images_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                mahalanobis_distances=mahalanobis_distances_reshaped
            )

    def export_using_query_images_input_ganomaly_gen_encoded_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        encoder_model,
        query_images,
        gen_encoded_logits
    ):
        """Adds generator encoded image outputs & beyond to export for GANomaly query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
            gen_encoded_logits: tensor, G(x) of shape
                (batch_size, latent_size).
        """
        # G(x).
        _, gen_encoded_images = generator_model(
            inputs=query_images, training=False
        )
        gen_encoded_images = tf.identity(
            input=gen_encoded_images,
            name="query_gen_encoded_images_{}".format(growth_idx)
        )

        if export_dict["export_query_gen_encoded_images"]:
            # shape = (batch_size, height, width, depth).
            # range = [-1., 1.].
            export_outputs.append(gen_encoded_images)

        if self.params["encoder"]["create"] and any(
            [
                export_dict["export_query_enc_encoded_logits"],
                export_dict["export_query_enc_encoded_images"]
            ]
        ):
            self.export_using_query_images_input_ganomaly_enc_encoded_logits_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                generator_model,
                encoder_model,
                gen_encoded_images
            )

        if any(
            [
                export_dict["export_query_anomaly_images_sigmoid"],
                export_dict["export_query_anomaly_images_linear"]
            ]
        ):
            self.export_using_query_images_input_ganomaly_anomaly_images(
                growth_idx,
                export_dict,
                export_outputs,
                query_images,
                gen_encoded_images
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
            self.export_using_query_images_input_ganomaly_mahalanobis_distances_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                query_images,
                gen_encoded_images
            )

    def export_using_query_images_input_ganomaly_gen_encoded_logits_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_outputs,
        generator_model,
        encoder_model,
        query_images
    ):
        """Adds generator encoded logit outputs & beyond to export for GANomaly query image inputs.

        Args:
            growth_idx: int, index of growth phase to export.
            export_dict: dict, user passed export parameters.
            export_outputs: list, stores tensors for export output.
            generator_model: `Model`, generator model network.
            encoder_model: `Model`, encoder model network.
            query_images: tensor, query image inputs of shape
                (batch_size, height, width, depth).
        """
        # Ge(x).
        gen_encoded_logits, _ = generator_model(
            inputs=query_images, training=False
        )
        gen_encoded_logits = tf.identity(
            input=gen_encoded_logits,
            name="query_gen_encoded_logits_{}".format(growth_idx)
        )

        if export_dict["export_query_gen_encoded_logits"]:
            # shape = (batch_size, 1).
            # range = (-inf, inf).
            export_outputs.append(gen_encoded_logits)

        if any(
            [
                export_dict["export_query_gen_encoded_images"],
                (
                    self.params["encoder"]["create"] and
                        any(
                            [
                                export_dict["export_query_enc_encoded_logits"],
                                export_dict["export_query_enc_encoded_images"]
                            ]
                        )
                ),
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
                )
            ]
        ):
            self.export_using_query_images_input_ganomaly_gen_encoded_images_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                generator_model,
                encoder_model,
                query_images,
                gen_encoded_logits
            )

    def export_using_query_images_input_ganomaly_query_images_and_beyond(
        self,
        growth_idx,
        export_dict,
        export_inputs_query_images,
        export_outputs,
        generator_model,
        encoder_model
    ):
        """Adds query image outputs & beyond to export for GANomaly query image inputs.

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
                generator_model.image_to_vector_input_layers[block_idx]
            )
        else:
            query_images = generator_model.inputs[0]

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

        if any(
            [
                export_dict["export_query_gen_encoded_logits"],
                export_dict["export_query_gen_encoded_images"],
                (
                    self.params["encoder"]["create"] and
                        any(
                            [
                                export_dict["export_query_enc_encoded_logits"],
                                export_dict["export_query_enc_encoded_images"]
                            ]
                        )
                ),
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
                )
            ]
        ):
            self.export_using_query_images_input_ganomaly_gen_encoded_logits_and_beyond(
                growth_idx,
                export_dict,
                export_outputs,
                generator_model,
                encoder_model,
                query_images
            )

    def export_using_query_images_input_ganomaly(
        self,
        growth_idx,
        export_dict,
        export_inputs_query_images,
        export_outputs,
        generator_model,
        encoder_model
    ):
        """Adds inputs and outputs to export for GANomaly query image inputs.

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
                export_dict["export_query_gen_encoded_logits"],
                export_dict["export_query_gen_encoded_images"],
                (
                    self.params["encoder"]["create"] and
                        any(
                            [
                                export_dict["export_query_enc_encoded_logits"],
                                export_dict["export_query_enc_encoded_images"]
                            ]
                        )
                ),
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
                )
            ]
        ):
            self.export_using_query_images_input_ganomaly_query_images_and_beyond(
                growth_idx,
                export_dict,
                export_inputs_query_images,
                export_outputs,
                generator_model,
                encoder_model
            )

    def create_serving_model_ganomaly(self, growth_idx):
        """Creates Keras `Model` for serving GANomaly paper.

        Args:
            growth_idx: int, index of growth phase to export.

        Returns:
            `tf.keras.Model` for serving.
        """
        export_dict = self.params["export"]
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

        self.export_using_query_images_input_ganomaly(
            growth_idx,
            export_dict,
            export_inputs_query_images,
            export_outputs,
            generator_model,
            encoder_model
        )

        if export_dict["export_all_growth_phases"]:
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
            inputs=export_inputs_query_images,
            outputs=export_outputs,
            name="serving_model_growth_{}_epoch_{}".format(
                growth_idx, self.epoch_idx
            )
        )

        if export_dict["print_serving_model_summaries"]:
            print("\nserving_model = {}".format(serving_model.summary()))

        return serving_model

    def create_serving_models_ganomaly(self):
        """Creates Keras `Model`s for serving for GANomaly paper.

        Args:
            growth_idx: int, index of growth phase to export.
        """
        if self.params["export"]["export_all_growth_phases"]:
            # Reset.
            self.serving_inputs_query_images_names_set.clear()
            self.serving_inputs_query_images.clear()
            self.serving_outputs.clear()
            for i in range(self.num_growths):
                self.create_serving_model_ganomaly(growth_idx=i)

            if self.params["export"]["print_serving_model_summaries"]:
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
                inputs=self.serving_inputs_query_images,
                outputs=self.serving_outputs,
                name="serving_model_all_growth_{}_epoch_{}".format(
                    self.growth_idx, self.epoch_idx
                )
            )
        else:
            self.serving_model = self.create_serving_model_ganomaly(
                growth_idx=self.growth_idx
            )
