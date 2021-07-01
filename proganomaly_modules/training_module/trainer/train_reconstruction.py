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


class TrainReconstruction(object):
    """Class used for training reconstruction model.
    """
    def __init__(self):
        """Instantiate instance of `TrainReconstruction`.
        """
        pass

    def resize_real_images(self, images):
        """Resizes real images to match the GAN's current size.

        Args:
            images: tensor, original images.

        Returns:
            Resized image tensor.
        """
        if self.params["training"]["reconstruction"]["use_multiple_resolution_records"]:
            return images

        height, width = self.params["generator"]["projection_dims"][0:2]
        resized_image = tf.image.resize(
            images=images,
            size=[
                height * (2 ** self.block_idx), width * (2 ** self.block_idx)
            ],
            method="nearest",
            name="resized_real_image_{}".format(self.growth_idx)
        )

        return resized_image

    def get_variables_and_gradients(self, loss, gradient_tape, scope):
        """Gets variables and gradients from model wrt. loss.

        Args:
            loss: tensor, shape of ().
            gradient_tape: instance of `GradientTape`.
            scope: str, the name of the network of interest.

        Returns:
            Lists of network's variables and gradients.
        """
        # Get trainable variables.
        variables = self.network_objects[scope].models[self.growth_idx].trainable_variables

        # Get gradients from gradient tape.
        gradients = gradient_tape.gradient(
            target=loss, sources=variables
        )

        # Clip gradients.
        if self.params[scope]["clip_gradients"]:
            gradients, _ = tf.clip_by_global_norm(
                t_list=gradients,
                clip_norm=self.params[scope]["clip_gradients"],
                name="{}_clip_by_global_norm_gradients".format(scope)
            )

        # Add variable names back in for identification.
        gradients = [
            tf.identity(
                input=g,
                name="{}_{}_gradients".format(scope, v.name[:-2])
            )
            if tf.is_tensor(x=g) else g
            for g, v in zip(gradients, variables)
        ]

        return variables, gradients

    def create_variable_and_gradient_histogram_summaries(
        self, variables, gradients, scope
    ):
        """Creates variable and gradient histogram summaries.

        Args:
            variables: list, network's trainable variables.
            gradients: list, gradients of network's trainable variables wrt.
                loss.
            scope: str, the name of the network of interest.
        """
        recon_dict = self.params["training"]["reconstruction"]
        if (
            recon_dict["write_variable_histogram_summaries"] or
            recon_dict["write_gradient_histogram_summaries"]
        ):
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=recon_dict["save_summary_steps"]
                        ), y=0
                    )
                ):
                    for v, g in zip(variables, gradients):
                        if recon_dict["write_variable_histogram_summaries"]:
                            tf.summary.histogram(
                                name="{}_variables/{}".format(
                                    scope, v.name[:-2]
                                ),
                                data=v,
                                step=self.global_step_var
                            )

                        if recon_dict["write_gradient_histogram_summaries"]:
                            if tf.is_tensor(x=g):
                                tf.summary.histogram(
                                    name="{}_gradients/{}".format(
                                        scope, v.name[:-2]
                                    ),
                                    data=g,
                                    step=self.global_step_var
                                )
                    self.summary_file_writer.flush()

    def get_network_losses_variables_and_gradients(self, real_images):
        """Gets losses, variables, & gradients for each network.

        Args:
            real_images: tensor, real images of shape
                (batch_size, height, width, depth).

        Returns:
            Dictionaries of network losses, variables, and gradients.
        """
        if self.params["generator"]["architecture"] == "berg":
            generator_encoder_loss_phase_fn = (
                self.generator_encoder_loss_phase_berg
            )
        elif self.params["generator"]["architecture"] == "GANomaly":
            generator_encoder_loss_phase_fn = (
                self.generator_encoder_loss_phase_ganomaly
            )

        discriminator_loss_dict = dict()
        with tf.GradientTape() as gen_tape, \
             tf.GradientTape() as enc_tape, \
             tf.GradientTape() as dis_tape:
            # Get fake logits from generator.
            (fake_images,
             generator_encoder_loss_dict) = generator_encoder_loss_phase_fn(
                real_images, training=True
            )

            if self.params["discriminator"]["create"]:
                # Get discriminator loss.
                discriminator_loss_dict = self.discriminator_loss_phase(
                    generator_encoder_loss_dict,
                    fake_images,
                    real_images,
                    training=True
                )

        # Create dicts to hold losses, variables, and gradients.
        loss_dict = {**generator_encoder_loss_dict, **discriminator_loss_dict}
        vars_dict = {}
        grads_dict = {}

        # Create dynamic lists to loop over depending on created networks.
        losses = [loss_dict["generator_total_loss"]]
        gradient_tapes = [gen_tape]
        scopes = ["generator"]

        if self.params["encoder"]["create"]:
            losses.append(loss_dict["encoder_total_loss"])
            gradient_tapes.append(enc_tape)
            scopes.append("encoder")

        if self.params["discriminator"]["create"]:
            losses.append(loss_dict["discriminator_total_loss"])
            gradient_tapes.append(dis_tape)
            scopes.append("discriminator")

        # Loop over generator and discriminator.
        for (loss, gradient_tape, scope_name) in zip(
            losses, gradient_tapes, scopes
        ):
            # Get variables and gradients from generator wrt. loss.
            variables, gradients = self.get_variables_and_gradients(
                loss, gradient_tape, scope_name
            )

            # Add variables and gradients to dictionaries.
            vars_dict[scope_name] = variables
            grads_dict[scope_name] = gradients

            # Create variable and gradient histogram summaries.
            self.create_variable_and_gradient_histogram_summaries(
                variables, gradients, scope_name
            )

        return loss_dict, vars_dict, grads_dict

    def train_network(self, variables, gradients, scope):
        """Trains network variables using gradients with optimizer.

        Args:
            variables: dict, lists for each network's trainable variables.
            gradients: dict, lists for each network's gradients of loss wrt
                network's trainable variables.
            scope: str, the name of the network of interest.
        """
        # Zip together gradients and variables.
        grads_and_vars = zip(gradients[scope], variables[scope])

        # Applying gradients to variables using optimizer.
        self.optimizers[scope].apply_gradients(
            grads_and_vars=(
                (g, v)
                for g,v in grads_and_vars
                if g is not None
            )
        )

    def get_losses_variables_and_gradients(self, features):
        """Gets losses, variables, and gradients.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Dictionaries of network losses, variables, and gradients.
        """
        # Extract real images from features dictionary.
        real_images = self.resize_real_images(images=features["image"])

        # Get gradients for training by running inputs through networks.
        losses, variables, gradients = (
            self.get_network_losses_variables_and_gradients(
                real_images
            )
        )

        return losses, variables, gradients

    def train_discriminator(self, features):
        """Trains discriminator network.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Dictionary of scalar losses for each network.
        """
        losses, variables, gradients = (
            self.get_losses_variables_and_gradients(features)
        )

        # Train discriminator network.
        if (
            self.params["discriminator"]["create"] and
            self.params["discriminator"]["train"]
        ):
            self.train_network(variables, gradients, scope="discriminator")

        return losses

    def train_generator_encoder(self, features):
        """Trains generator and encoder networks jointly.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Dictionary of scalar losses for each network.
        """
        losses, variables, gradients = (
            self.get_losses_variables_and_gradients(features)
        )

        # Train generator network.
        if self.params["generator"]["train"]:
            self.train_network(variables, gradients, scope="generator")

        # Train encoder network.
        if (
            self.params["encoder"]["create"] and
            self.params["encoder"]["train"]
        ):
            self.train_network(variables, gradients, scope="encoder")

        return losses
