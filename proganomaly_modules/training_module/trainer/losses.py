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

from . import losses_berg
from . import losses_ganomaly


class Losses(
    losses_berg.LossesBerg,
    losses_ganomaly.LossesGanomaly
):
    """Class used for both training & evaluation losses.
    """
    def __init__(self):
        """Instantiate instance of `Losses`.
        """
        pass

    def get_fake_logits_loss(self, fake_image_type, fake_logits):
        """Gets fake logits loss.

        Args:
            fake_image_type: str, the type of fake image.
            fake_logits: tensor, shape of (batch_size, 1).

        Returns:
            Tensor of fake logit's loss of shape ().
        """
        if self.params["training"]["distribution_strategy"]:
            # Calculate base generator loss.
            fake_logits_loss = tf.nn.compute_average_loss(
                per_example_loss=fake_logits,
                global_batch_size=(
                    self.global_batch_size_schedule_reconstruction[self.block_idx]
                )
            )
        else:
            # Calculate base generator loss.
            fake_logits_loss = tf.reduce_mean(
                input_tensor=fake_logits,
                name="fake_logits_loss"
            )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ),
                        y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/{}_loss".format(fake_image_type),
                        data=fake_logits_loss,
                        step=self.global_step_var
                    )
                    self.summary_file_writer.flush()

        return fake_logits_loss

    def generator_loss_phase(self, fake_image_type, fake_images, training):
        """Gets logits and loss for generator.

        Args:
            fake_image_type: str, the type of fake image.
            fake_images: tensor, generated images of shape
                (batch_size, iamge_height, image_width, image_depth).
            training: bool, if model should be training.

        Returns:
            Generator loss tensor of shape ().
        """
        if self.params["training"]["reconstruction"]["write_generator_image_summaries"] and training:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ),
                        y=0
                    )
                ):
                    tf.summary.image(
                        name="fake_images_{}".format(fake_image_type),
                        data=fake_images,
                        step=self.global_step_var,
                        max_outputs=5
                    )
                    self.summary_file_writer.flush()

        # Get fake logits from discriminator using generator's output image.
        fake_discriminator_logits = (
            self.network_objects["discriminator"].models[self.growth_idx](
                inputs=fake_images, training=training
            )
        )

        # Get generator loss from discriminator.
        generator_loss = self.get_fake_logits_loss(
            fake_image_type=fake_image_type,
            fake_logits=fake_discriminator_logits
        )

        return generator_loss

    def get_encoder_l1_z_loss(self, z, z_hat):
        """Gets encoder L1 latent vector loss.

        Args:
            z: tensor, latent vector of shape
                (batch_size, generator_latent_size).
            z_hat: tensor, latent vector of shape
                (batch_size, generator_latent_size).

        Returns:
            Tensor of encoder's L2 image loss of shape ().
        """
        # Get difference between z and z-hat.
        z_diff = tf.subtract(x=z, y=z_hat, name="z_diff")

        # Get L1 norm of latent vector difference.
        if self.params["training"]["reconstruction"]["normalize_reconstruction_losses"]:
            z_diff_l1_norm = tf.reduce_mean(
                input_tensor=tf.abs(x=z_diff),
                axis=-1,
                name="z_diff_l1_norm"
            ) + 1e-8
        else:
            z_diff_l1_norm = tf.reduce_sum(
                input_tensor=tf.abs(x=z_diff),
                axis=[-1],
                name="z_diff_l1_norm"
            ) + 1e-8

        if self.params["training"]["distribution_strategy"]:
            # Calculate base encoder loss.
            encoder_z_loss = tf.nn.compute_average_loss(
                per_example_loss=z_diff_l1_norm,
                global_batch_size=(
                    self.global_batch_size_schedule_reconstruction[self.block_idx]
                )
            )
        else:
            # Calculate base encoder loss.
            encoder_z_loss = tf.reduce_mean(
                input_tensor=z_diff_l1_norm,
                name="encoder_l1_z_loss"
            )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/encoder_l1_z_loss",
                        data=encoder_z_loss,
                        step=self.global_step_var
                    )
                    self.summary_file_writer.flush()

        return encoder_z_loss

    def get_encoder_l2_z_loss(self, z, z_hat):
        """Gets encoder L2 latent vector loss.

        Args:
            z: tensor, latent vector of shape
                (batch_size, generator_latent_size).
            z_hat: tensor, latent vector of shape
                (batch_size, generator_latent_size).

        Returns:
            Tensor of encoder's L2 image loss of shape [].
        """
        # Get difference between z and z-hat.
        z_diff = tf.subtract(x=z, y=z_hat, name="z_diff")

        # Get L2 norm of latent vector difference.
        if self.params["training"]["reconstruction"]["normalize_reconstruction_losses"]:
            z_diff_l2_norm = tf.reduce_mean(
                input_tensor=tf.square(x=z_diff),
                axis=-1,
                name="z_diff_l2_norm"
            )
        else:
            z_diff_l2_norm = tf.reduce_sum(
                input_tensor=tf.square(x=z_diff),
                axis=-1,
                name="z_diff_l2_norm"
            )

        if self.params["training"]["distribution_strategy"]:
            # Calculate base encoder loss.
            encoder_z_loss = tf.nn.compute_average_loss(
                per_example_loss=z_diff_l2_norm,
                global_batch_size=(
                    self.global_batch_size_schedule_reconstruction[self.block_idx]
                )
            )
        else:
            # Calculate base encoder loss.
            encoder_z_loss = tf.reduce_mean(
                input_tensor=z_diff_l2_norm,
                name="encoder_l2_z_loss"
            )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/encoder_l2_z_loss",
                        data=encoder_z_loss,
                        step=self.global_step_var
                    )
                    self.summary_file_writer.flush()

        return encoder_z_loss

    def get_encoder_l1_image_loss(self, images, encoded_images):
        """Gets encoder L1 image loss.

        Args:
            images: tensor, either real images or images generated by the
                generator from random noise. Shape of
                (batch_size, image_height, image_width, depth).
            encoded_images: tensor, images generated by the generator from
                encoder's vector output of shape
                (batch_size, image_height, image_width, depth).

        Returns:
            Tensor of encoder's L1 image loss of shape ().
        """
        # Get difference between fake images and encoder images.
        generator_encoder_image_diff = tf.subtract(
            x=images,
            y=encoded_images,
            name="generator_encoder_image_diff"
        )

        # Get L1 norm of image difference.
        if self.params["training"]["reconstruction"]["normalize_reconstruction_losses"]:
            image_diff_l1_norm = tf.reduce_mean(
                input_tensor=tf.abs(x=generator_encoder_image_diff),
                axis=[1, 2, 3],
                name="image_diff_l1_norm"
            ) + 1e-8
        else:
            image_diff_l1_norm = tf.reduce_sum(
                input_tensor=tf.abs(x=generator_encoder_image_diff),
                axis=[1, 2, 3],
                name="image_diff_l1_norm"
            ) + 1e-8

        if self.params["training"]["distribution_strategy"]:
            # Calculate base encoder loss.
            encoder_image_loss = tf.nn.compute_average_loss(
                per_example_loss=image_diff_l1_norm,
                global_batch_size=(
                    self.global_batch_size_schedule_reconstruction[self.block_idx]
                )
            )
        else:
            # Calculate base encoder loss.
            encoder_image_loss = tf.reduce_mean(
                input_tensor=image_diff_l1_norm,
                name="encoder_l1_image_loss"
            )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/encoder_l1_image_loss",
                        data=encoder_image_loss,
                        step=self.global_step_var
                    )
                    self.summary_file_writer.flush()

        return encoder_image_loss

    def get_encoder_l2_image_loss(self, images, encoded_images):
        """Gets encoder L2 image loss.

        Args:
            images: tensor, either real images or images generated by the
                generator from random noise. Shape of
                (batch_size, image_height, image_width, depth).
            encoded_images: tensor, images generated by the generator from
                encoder's vector output of shape
                (batch_size, image_height, image_width, depth).

        Returns:
            Tensor of encoder's L2 image loss of shape ().
        """
        # Get difference between fake images and encoder images.
        generator_encoder_image_diff = tf.subtract(
            x=images,
            y=encoded_images,
            name="generator_encoder_image_diff"
        )

        # Get L2 norm of image difference.
        if self.params["training"]["reconstruction"]["normalize_reconstruction_losses"]:
            image_diff_l2_norm = tf.reduce_mean(
                input_tensor=tf.square(x=generator_encoder_image_diff),
                axis=[1, 2, 3],
                name="image_diff_l2_norm"
            )
        else:
            image_diff_l2_norm = tf.reduce_sum(
                input_tensor=tf.square(x=generator_encoder_image_diff),
                axis=[1, 2, 3],
                name="image_diff_l2_norm"
            )

        if self.params["training"]["distribution_strategy"]:
            # Calculate base encoder loss.
            encoder_image_loss = tf.nn.compute_average_loss(
                per_example_loss=image_diff_l2_norm,
                global_batch_size=(
                    self.global_batch_size_schedule_reconstruction[self.block_idx]
                )
            )
        else:
            # Calculate base encoder loss.
            encoder_image_loss = tf.reduce_mean(
                input_tensor=image_diff_l2_norm,
                name="encoder_l2_image_loss"
            )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/encoder_l2_image_loss",
                        data=encoder_image_loss,
                        step=self.global_step_var
                    )
                    self.summary_file_writer.flush()

        return encoder_image_loss

    def get_network_regularization_loss(self, network):
        """Gets network's regularization loss.

        Args:
            network: str, name of network model.

        Returns:
            Tensor of network's regularization loss of shape ().
        """
        if self.params["training"]["distribution_strategy"]:
            # Get regularization losses.
            reg_loss = tf.nn.scale_regularization_loss(
                regularization_loss=sum(
                    self.network_objects[network].models[self.growth_idx].losses
                )
            )
        else:
            # Get regularization losses.
            reg_loss = sum(self.network_objects[network].models[self.growth_idx].losses)

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=global_step,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/{}_reg_loss".format(network),
                        data=reg_loss,
                        step=global_step
                    )

        return reg_loss

    def get_discriminator_loss_real_image_losses(self, real_images, training):
        """Gets real image losses for discriminator.

        Args:
            real_images: tensor, real images from input of shape
                (batch_size, image_height, image_width, depth).
            training: bool, if in training mode.

        Returns:
            Dictionary of scalar losses and running loss scalar tensor of
                discriminator.
        """
        dis_loss_weights = self.params["discriminator"]["losses"]

        # Create empty dict for unweighted losses.
        loss_dict = {}

        discriminator_real_loss = tf.zeros(shape=(), dtype=tf.float32)
        if dis_loss_weights["D_of_x_loss_weight"]:
            # Get real logits from discriminator using real image.
            real_logits = self.network_objects["discriminator"].models[self.growth_idx](
                inputs=real_images, training=training
            )

            if self.params["training"]["distribution_strategy"]:
                discriminator_real_loss = tf.nn.compute_average_loss(
                    per_example_loss=real_logits,
                    global_batch_size=(
                        self.global_batch_size_schedule_reconstruction[self.block_idx]
                    )
                )
            else:
                discriminator_real_loss = tf.reduce_mean(
                    input_tensor=real_logits,
                    name="real_loss"
                )
            loss_dict["D(x)"] = discriminator_real_loss

            discriminator_real_loss = tf.multiply(
                x=dis_loss_weights["D_of_x_loss_weight"],
                y=discriminator_real_loss
            )

            # Get discriminator epsilon drift penalty.
            if self.params["discriminator"]["epsilon_drift"]:
                epsilon_drift_penalty = tf.multiply(
                    x=self.params["discriminator"]["epsilon_drift"],
                    y=tf.reduce_mean(input_tensor=tf.square(x=real_logits)),
                    name="epsilon_drift_penalty"
                )
                loss_dict["epsilon_drift_penalty"] = epsilon_drift_penalty

            if self.params["training"]["reconstruction"]["write_loss_summaries"]:
                # Add summaries for TensorBoard.
                with self.summary_file_writer.as_default():
                    with tf.summary.record_if(
                        condition=tf.equal(
                            x=tf.math.floormod(
                                x=self.global_step_var,
                                y=self.params["training"]["reconstruction"]["save_summary_steps"]
                            ), y=0
                        )
                    ):
                        tf.summary.scalar(
                            name="losses/discriminator_real_loss",
                            data=discriminator_real_loss,
                            step=self.global_step_var
                        )
                        if self.params["discriminator"]["epsilon_drift"]:
                            tf.summary.scalar(
                                name="losses/epsilon_drift_penalty",
                                data=epsilon_drift_penalty,
                                step=self.global_step_var
                            )
                        self.summary_file_writer.flush()

        return loss_dict, discriminator_real_loss

    def _get_gradient_penalty_loss(self, fake_images, real_images):
        """Gets discriminator gradient penalty loss.

        Args:
            fake_images: tensor, images generated by the generator from random
                noise of shape (batch_size, image_size, image_size, 3).
            real_images: tensor, real images from input of shape
                (batch_size, image_height, image_width, 3).

        Returns:
            Discriminator's gradient penalty loss of shape ().
        """
        batch_size = real_images.shape[0]

        # Get a random uniform number rank 4 tensor.
        random_uniform_num = tf.random.uniform(
            shape=(batch_size, 1, 1, 1),
            minval=0., maxval=1.,
            dtype=tf.float32,
            name="gp_random_uniform_num"
        )

        # Find the element-wise difference between images.
        image_difference = fake_images - real_images

        # Get random samples from this mixed image distribution.
        mixed_images = random_uniform_num * image_difference
        mixed_images += real_images

        # Get loss from interpolated mixed images and watch for gradients.
        with tf.GradientTape() as gp_tape:
            # Watch interpolated mixed images.
            gp_tape.watch(tensor=mixed_images)

            # Send to the discriminator to get logits.
            mixed_logits = self.network_objects["discriminator"].models[self.growth_idx](
                inputs=mixed_images, training=True
            )

            # Get the mixed loss.
            mixed_loss = tf.reduce_sum(
                input_tensor=mixed_logits,
                name="gp_mixed_loss"
            )

        # Get gradient from returned list of length 1.
        mixed_gradients = gp_tape.gradient(
            target=mixed_loss, sources=[mixed_images]
        )[0]

        # Get gradient's L2 norm.
        mixed_norms = tf.sqrt(
            x=tf.reduce_sum(
                input_tensor=tf.square(
                    x=mixed_gradients,
                    name="gp_squared_grads"
                ),
                axis=[1, 2, 3]
            ) + 1e-8
        )

        # Get squared difference from target of 1.0.
        squared_difference = tf.square(
            x=tf.math.subtract(
                x=mixed_norms,
                y=self.params["discriminator"]["gradient_penalty_target"]
            ),
            name="gp_squared_difference"
        )

        # Get gradient penalty scalar.
        gradient_penalty = tf.reduce_mean(
            input_tensor=squared_difference, name="gp_gradient_penalty"
        )

        # Multiply with lambda to get gradient penalty loss.
        gradient_penalty_loss = tf.multiply(
            x=tf.divide(
                x=self.params["discriminator"]["gradient_penalty_coefficient"],
                y=tf.square(
                    x=self.params["discriminator"]["gradient_penalty_target"]
                )
            ),
            y=gradient_penalty,
            name="gp_gradient_penalty_loss"
        )

        return gradient_penalty_loss

    def get_discriminator_loss(
        self,
        generator_encoder_loss_dict,
        discriminator_loss_dict,
        discriminator_real_loss,
        fake_weight_type,
        fake_image_type,
        fake_images,
        real_images
    ):
        """Gets final discriminator loss.

        Args:
            generator_encoder_loss_dict: dict, scalar losses from
                generator/encoder loss phase.
            discriminator_loss_dict: dict, scalar losses from discriminator.
            discriminator_real_loss: tensor, scalar loss for discriminator for
                real images.
            fake_weight_type: str, the type of fake weight parameter.
            fake_image_type: str, the type of fake image.
            fake_images: tensor, images generated by the generator of shape
                (batch_size, image_size, image_size, image_depth).
            real_images: tensor, real images from input of shape
                (batch_size, image_height, image_width, image_depth).
        """
        dis_loss_weights_berg = self.params["discriminator"]["losses"]["berg"]
        dis_loss_weights_ganomaly = (
            self.params["discriminator"]["losses"]["GANomaly"]
        )
        fake_loss_weight_berg = dis_loss_weights_berg.get(
            "D_of_{}_loss_weight".format(fake_weight_type)
        )
        fake_loss_weight_ganomaly = dis_loss_weights_ganomaly.get(
            "D_of_{}_loss_weight".format(fake_weight_type)
        )
        fake_loss_weight = (
            fake_loss_weight_berg
            if fake_loss_weight_berg is not None
            else fake_loss_weight_ganomaly
        )
        discriminator_fake_loss = tf.multiply(
            x=fake_loss_weight,
            y=generator_encoder_loss_dict.get(
                fake_image_type, tf.zeros(shape=(), dtype=tf.float32)
            )
        )

        discriminator_loss = tf.subtract(
            x=discriminator_fake_loss,
            y=discriminator_real_loss,
            name="discriminator_loss"
        )
        discriminator_loss_dict["{}-D(x)".format(fake_image_type)] = (
            discriminator_loss
        )

        # Get discriminator gradient penalty loss.
        discriminator_gradient_penalty = self._get_gradient_penalty_loss(
            fake_images, real_images
        )
        discriminator_loss_dict["{}_gradient_penalty".format(fake_image_type)] = (
            discriminator_gradient_penalty
        )

        epsilon_drift_penalty = discriminator_loss_dict.get(
            "epsilon_drift_penalty", tf.zeros(shape=(), dtype=tf.float32)
        )

        # Get discriminator Wasserstein GP loss.
        discriminator_wasserstein_gp_loss = tf.add_n(
            inputs=[
                discriminator_loss,
                discriminator_gradient_penalty,
                epsilon_drift_penalty
            ],
            name="discriminator_wasserstein_gp_loss"
        )
        discriminator_loss_dict["{}_wgan_gp".format(fake_image_type)] = (
            discriminator_wasserstein_gp_loss
        )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step_var,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/discriminator_{}_fake_loss".format(
                            fake_image_type
                        ),
                        data=discriminator_fake_loss,
                        step=self.global_step_var
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_{}_loss".format(
                            fake_image_type
                        ),
                        data=discriminator_loss,
                        step=self.global_step_var
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_{}_gradient_penalty".format(
                            fake_image_type
                        ),
                        data=discriminator_gradient_penalty,
                        step=self.global_step_var
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_{}_wasserstein_gp_loss".format(
                            fake_image_type
                        ),
                        data=discriminator_wasserstein_gp_loss,
                        step=self.global_step_var
                    )
                    self.summary_file_writer.flush()

    def discriminator_loss_phase(
        self, generator_encoder_loss_dict, fake_images, real_images, training
    ):
        """Gets real logits and loss for discriminator.

        Args:
            generator_encoder_loss_dict: dict, scalar losses from
                generator/encoder loss phase.
            fake_images: dict, image tensors generated by the generator from
                random noise of shape
                (batch_size, image_size, image_size, depth).
            real_images: tensor, real images from input of shape
                (batch_size, image_height, image_width, depth).
            training: bool, if in training mode.

        Returns:
            Dictionary of scalar losses.
        """
        # Loss params-name map.
        loss_params_name_map = dict()
        if self.params["generator"]["architecture"] == "berg":
            loss_params_name_map = {
                "G_of_z": "D(G(z))",
                "G_of_E_of_x": "D(G(E(x)))",
                "G_of_E_of_G_of_z": "D(G(E(G(z))))"
            }
        elif self.params["generator"]["architecture"] == "GANomaly":
            loss_params_name_map = {
                "G_of_x": "D(G(x))"
            }

        (discriminator_loss_dict,
         discriminator_real_loss
        ) = self.get_discriminator_loss_real_image_losses(
            real_images, training
        )

        # Get discriminator losses.
        discriminator_losses = []
        for key in fake_images.keys():
            self.get_discriminator_loss(
                generator_encoder_loss_dict=generator_encoder_loss_dict,
                discriminator_loss_dict=discriminator_loss_dict,
                discriminator_real_loss=discriminator_real_loss,
                fake_weight_type=key,
                fake_image_type=loss_params_name_map[key],
                fake_images=fake_images[key],
                real_images=real_images
            )

            discriminator_losses.append(
                discriminator_loss_dict[
                    "{}_wgan_gp".format(loss_params_name_map[key])
                ]
            )

        # Combine losses into discriminator total loss.
        discriminator_reg_loss = self.get_network_regularization_loss(
            network="discriminator"
        )
        discriminator_loss_dict["discriminator_reg_loss"] = (
            discriminator_reg_loss
        )
        discriminator_total_loss = discriminator_reg_loss

        discriminator_total_loss += tf.reduce_sum(
            input_tensor=discriminator_losses
        )
        discriminator_loss_dict["discriminator_total_loss"] = (
            discriminator_total_loss
        )

        if self.params["training"]["reconstruction"]["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=global_step,
                            y=self.params["training"]["reconstruction"]["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="optimized_losses/discriminator_total_loss",
                        data=discriminator_total_loss,
                        step=global_step
                    )
                    summary_file_writer.flush()

        return discriminator_loss_dict
