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


class LossesBerg(object):
    """Class used for both training & evaluation losses for Berg paper.
    """
    def __init__(self):
        """Instantiate instance of `LossesBerg`.
        """
        pass

    def encoder_loss_phase_berg(
        self, Z, fake_images_from_z, real_images, training
    ):
        """Gets losses for encoder.

        Args:
            Z: tensor, latent vector of shape
                (batch_size, generator_latent_size).
            fake_images_from_z: tensor, images of shape
                (batch_size, image_height, image_width, depth).
            real_images: tensor, images of shape
                (batch_size, image_height, image_width, depth).
            training: bool, if model should be training.

        Returns:
            3-tuple of encoder loss tensors of shape () for fake z, fake
                image, real z, and real image cases, respectively, and
                dictionary of encoder losses.
        """
        gen_loss_weights = self.params["generator"]["losses"]["berg"]
        enc_loss_weights = self.params["encoder"]["losses"]["berg"]
        recon_dict = self.params["training"]["reconstruction"]
        fake_images_from_z_hat_from_real_images = None
        fake_images_from_z_hat_from_fake_images_from_z = None
        encoder_loss_dict = {}

        if any(
            [
                gen_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"],
                gen_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"],
                gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"],
                enc_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"],
                enc_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"],
                enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"]
            ]
        ):
            # E(G(Z))
            z_hat_from_fake_images_from_z = (
                self.network_objects["encoder"].models[self.growth_idx](
                    inputs=fake_images_from_z, training=training
                )
            )

            if (
                gen_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"] or
                enc_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"]
            ):
                encoder_loss_dict["z-E(G(z))_L1"] = self.get_encoder_l1_z_loss(
                    z=Z, z_hat=z_hat_from_fake_images_from_z
                )

            if (
                gen_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"] or
                enc_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"]
            ):
                encoder_loss_dict["z-E(G(z))_L2"] = self.get_encoder_l2_z_loss(
                    z=Z, z_hat=z_hat_from_fake_images_from_z
                )

            if any(
                [
                    gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                    gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"],
                    enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                    enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"]
                ]
            ):
                # G(E(G(Z)))
                fake_images_from_z_hat_from_fake_images_from_z = (
                    self.network_objects["generator"].models[self.growth_idx](
                        inputs=z_hat_from_fake_images_from_z,
                        training=training
                    )
                )

                if (
                    self.params["generator"]["add_uniform_noise_to_fake_images"] and
                    training
                ):
                    fake_images_from_z_hat_from_fake_images_from_z += (
                        tf.random.uniform(
                            shape=tf.shape(
                                input=fake_images_from_z_hat_from_fake_images_from_z
                            )
                        )
                    )

                if (
                    gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"] or
                    enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"]
                ):
                    encoder_loss_dict["G(z)-G(E(G(z)))_L1"] = (
                        self.get_encoder_l1_image_loss(
                            images=fake_images_from_z,
                            encoded_images=(
                                fake_images_from_z_hat_from_fake_images_from_z
                            )
                        )
                    )

                if (
                    gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"] or
                    enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"]
                ):
                    encoder_loss_dict["G(z)-G(E(G(z)))_L2"] = (
                        self.get_encoder_l2_image_loss(
                            images=fake_images_from_z,
                            encoded_images=(
                                fake_images_from_z_hat_from_fake_images_from_z
                            )
                        )
                    )

                if recon_dict["write_encoder_image_summaries"] and training:
                    # Add summaries for TensorBoard.
                    with self.summary_file_writer.as_default():
                        with tf.summary.record_if(
                            condition=tf.equal(
                                x=tf.math.floormod(
                                    x=self.global_step_var,
                                    y=recon_dict["save_summary_steps"]
                                ),
                                y=0
                            )
                        ):
                            tf.summary.image(
                                name="fake_images_from_z_hat_from_fake_images_from_z",
                                data=fake_images_from_z_hat_from_fake_images_from_z,
                                step=self.global_step_var,
                                max_outputs=5
                            )
                            self.summary_file_writer.flush()

        if any(
            [
                gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"],
                gen_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"],
                gen_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"],
                enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"],
                enc_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"],
                enc_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"]
            ]
        ):
            # E(x)
            z_hat_from_real_images = (
                self.network_objects["encoder"].models[self.growth_idx](
                    inputs=real_images, training=training
                )
            )

            # G(E(x))
            fake_images_from_z_hat_from_real_images = (
                self.network_objects["generator"].models[self.growth_idx](
                    inputs=z_hat_from_real_images, training=training
                )
            )

            if (
                self.params["generator"]["add_uniform_noise_to_fake_images"] and
                training
            ):
                fake_images_from_z_hat_from_real_images += (
                    tf.random.uniform(
                        shape=tf.shape(
                            input=fake_images_from_z_hat_from_real_images
                        )
                    )
                )

            if recon_dict["write_encoder_image_summaries"] and training:
                # Add summaries for TensorBoard.
                with self.summary_file_writer.as_default():
                    with tf.summary.record_if(
                        condition=tf.equal(
                            x=tf.math.floormod(
                                x=self.global_step_var,
                                y=recon_dict["save_summary_steps"]
                            ),
                            y=0
                        )
                    ):
                        tf.summary.image(
                            name="fake_images_from_z_hat_from_real_images",
                            data=fake_images_from_z_hat_from_real_images,
                            step=self.global_step_var,
                            max_outputs=5
                        )
                        self.summary_file_writer.flush()

            if any(
                [
                    gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                    gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"],
                    enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                    enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"]
                ]
            ):
                # E(G(E(x)))
                z_hat_from_fake_images_from_z_hat_from_real_images = (
                    self.network_objects["encoder"].models[self.growth_idx](
                        inputs=fake_images_from_z_hat_from_real_images,
                        training=training
                    )
                )

                if (
                    gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"] or
                    enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"]
                ):
                    encoder_loss_dict["E(x)-E(G(E(x)))_L1"] = self.get_encoder_l1_z_loss(
                        z=z_hat_from_real_images,
                        z_hat=z_hat_from_fake_images_from_z_hat_from_real_images
                    )

                if (
                    gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"] or
                    enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"]
                ):
                    encoder_loss_dict["E(x)-E(G(E(x)))_L2"] = self.get_encoder_l2_z_loss(
                        z=z_hat_from_real_images,
                        z_hat=z_hat_from_fake_images_from_z_hat_from_real_images
                    )

            if (
                gen_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"] or
                enc_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"]
            ):
                encoder_loss_dict["x-G(E(x))_L1"] = (
                    self.get_encoder_l1_image_loss(
                        images=real_images,
                        encoded_images=fake_images_from_z_hat_from_real_images
                    )
                )

            if (
                gen_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"] or
                enc_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"]
            ):
                encoder_loss_dict["x-G(E(x))_L2"] = (
                    self.get_encoder_l2_image_loss(
                        images=real_images,
                        encoded_images=fake_images_from_z_hat_from_real_images
                    )
                )

        return (
            fake_images_from_z_hat_from_real_images,
            fake_images_from_z_hat_from_fake_images_from_z,
            encoder_loss_dict
        )

    def generator_encoder_loss_phase_berg(self, real_images, training):
        """Gets fake logits and loss for generator.

        Args:
            real_images: tensor, real images from input of shape
                (batch_size, image_height, image_width, depth).
            training: bool, if model should be training.

        Returns:
            Dictionary of fake image tensors each of shape
                (batch_size, iamge_height, image_width, image_depth),
                dictionary of fake logits tensors each of shape
                (batch_size, 1), and dictionary of scalar losses.
        """
        gen_loss_weights = self.params["generator"]["losses"]["berg"]
        enc_loss_weights = self.params["encoder"]["losses"]["berg"]
        dis_loss_weights = self.params["discriminator"]["losses"]["berg"]
        recon_dict = self.params["training"]["reconstruction"]

        # Create dictionary to contain all raw losses.
        loss_dict = {}

        # Generator phase.
        block_idx = (self.growth_idx + 1) // 2
        batch_size = (
            recon_dict["train_batch_size_schedule"][block_idx]
            if training
            else recon_dict["eval_batch_size_schedule"][block_idx]
        )

        # Create random noise latent vector for each batch example.
        Z = tf.random.normal(
            shape=(batch_size, self.params["generator"]["latent_size"]),
            mean=self.params["generator"]["berg"]["latent_mean"],
            stddev=self.params["generator"]["berg"]["latent_stddev"],
            dtype=tf.float32
        )

        # Get generated image from generator network from gaussian noise.
        # G(z)
        fake_images_from_z = self.network_objects["generator"].models[self.growth_idx](
            inputs=Z, training=training
        )

        if (
            self.params["generator"]["add_uniform_noise_to_fake_images"] and
            training
        ):
            fake_images_from_z += tf.random.uniform(
                shape=tf.shape(input=fake_images_from_z)
            )

        # Encoder phase.
        encoder_loss_dict = dict()
        if self.params["encoder"]["create"]:
            (fake_images_from_z_hat_from_real_images,
             fake_images_from_z_hat_from_fake_images_from_z,
             encoder_loss_dict) = self.encoder_loss_phase_berg(
                Z, fake_images_from_z, real_images, training
            )

        # Add any encoder losses to loss dict.
        loss_dict.update(encoder_loss_dict)

        # Discriminator phase.
        if self.params["discriminator"]["create"]:
            if any(
                [
                    gen_loss_weights["D_of_G_of_z_loss_weight"],
                    dis_loss_weights["D_of_G_of_z_loss_weight"]
                ]
            ):
                # D(G(z))
                loss_dict["D(G(z))"] = self.generator_loss_phase(
                    fake_image_type="D(G(z))",
                    fake_images=fake_images_from_z,
                    training=training
                )

            if any(
                [
                    gen_loss_weights["D_of_G_of_E_of_x_loss_weight"],
                    enc_loss_weights["D_of_G_of_E_of_x_loss_weight"],
                    dis_loss_weights["D_of_G_of_E_of_x_loss_weight"]
                ]
            ):
                # D(G(E(x)))
                loss_dict["D(G(E(x)))"] = self.generator_loss_phase(
                    fake_image_type="D(G(E(x)))",
                    fake_images=fake_images_from_z_hat_from_real_images,
                    training=training
                )

            if any(
                [
                    gen_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"],
                    enc_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"],
                    dis_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"],
                ]
            ):
                # D(G(E(G(z))))
                loss_dict["D(G(E(G(z))))"] = self.generator_loss_phase(
                    fake_image_type="D(G(E(G(z))))",
                    fake_images=fake_images_from_z_hat_from_fake_images_from_z,
                    training=training
                )

        # Combine losses into generator total loss.
        generator_reg_loss = self.get_network_regularization_loss(
            network="generator"
        )
        loss_dict["generator_reg_loss"] = generator_reg_loss
        generator_total_loss = generator_reg_loss

        if self.params["discriminator"]["create"]:
            if gen_loss_weights["D_of_G_of_z_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["D_of_G_of_z_loss_weight"],
                    y=-loss_dict["D(G(z))"]
                )

            if gen_loss_weights["D_of_G_of_E_of_x_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["D_of_G_of_E_of_x_loss_weight"],
                    y=-loss_dict["D(G(E(x)))"]
                )

            if gen_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"],
                    y=-loss_dict["D(G(E(G(z))))"]
                )

        if self.params["encoder"]["create"]:
            if gen_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"],
                    y=loss_dict["z-E(G(z))_L1"]
                )

            if gen_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"],
                    y=loss_dict["z-E(G(z))_L2"]
                )

            if gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                    y=loss_dict["G(z)-G(E(G(z)))_L1"]
                )

            if gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"],
                    y=loss_dict["G(z)-G(E(G(z)))_L2"]
                )

            if gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                    y=loss_dict["E(x)-E(G(E(x)))_L1"]
                )

            if gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"],
                    y=loss_dict["E(x)-E(G(E(x)))_L2"]
                )

            if gen_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"],
                    y=loss_dict["x-G(E(x))_L1"]
                )

            if gen_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"],
                    y=loss_dict["x-G(E(x))_L2"]
                )

        loss_dict["generator_total_loss"] = generator_total_loss

        if self.params["encoder"]["create"]:
            # Combine losses into encoder total loss.
            encoder_reg_loss = self.get_network_regularization_loss(
                network="encoder"
            )
            loss_dict["encoder_reg_loss"] = encoder_reg_loss
            encoder_total_loss = encoder_reg_loss

            if self.params["discriminator"]["create"]:
                if enc_loss_weights["D_of_G_of_E_of_x_loss_weight"]:
                    encoder_total_loss += tf.multiply(
                        x=enc_loss_weights["D_of_G_of_E_of_x_loss_weight"],
                        y=-loss_dict["D(G(E(x)))"]
                    )

                if enc_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"]:
                    encoder_total_loss += tf.multiply(
                        x=enc_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"],
                        y=-loss_dict["D(G(E(G(z))))"]
                    )

            if enc_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["z_minus_E_of_G_of_z_l1_loss_weight"],
                    y=loss_dict["z-E(G(z))_L1"]
                )

            if enc_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["z_minus_E_of_G_of_z_l2_loss_weight"],
                    y=loss_dict["z-E(G(z))_L2"]
                )

            if enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                    y=loss_dict["G(z)-G(E(G(z)))_L1"]
                )

            if enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"],
                    y=loss_dict["G(z)-G(E(G(z)))_L2"]
                )

            if enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                    y=loss_dict["E(x)-E(G(E(x)))_L1"]
                )

            if enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"],
                    y=loss_dict["E(x)-E(G(E(x)))_L2"]
                )

            if enc_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["x_minus_G_of_E_of_x_l1_loss_weight"],
                    y=loss_dict["x-G(E(x))_L1"]
                )

            if enc_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["x_minus_G_of_E_of_x_l2_loss_weight"],
                    y=loss_dict["x-G(E(x))_L2"]
                )

            loss_dict["encoder_total_loss"] = encoder_total_loss

        if recon_dict["write_loss_summaries"]:
            # Add summaries for TensorBoard.
            with summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=global_step,
                            y=recon_dict["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="optimized_losses/generator_total_loss",
                        data=generator_total_loss,
                        step=global_step
                    )
                    if self.params["encoder"]["create"]:
                        tf.summary.scalar(
                            name="optimized_losses/encoder_total_loss",
                            data=encoder_total_loss,
                            step=global_step
                        )
                    summary_file_writer.flush()

        # Create dicts of chosen fake images & logits for discriminator later.
        fake_images = dict()
        if self.params["discriminator"]["create"]:
            if dis_loss_weights["D_of_G_of_z_loss_weight"]:
                fake_images["G_of_z"] = fake_images_from_z

            if dis_loss_weights["D_of_G_of_E_of_x_loss_weight"]:
                fake_images["G_of_E_of_x"] = (
                    fake_images_from_z_hat_from_real_images
                )

            if dis_loss_weights["D_of_G_of_E_of_G_of_z_loss_weight"]:
                fake_images["G_of_E_of_G_of_z"] = (
                    fake_images_from_z_hat_from_fake_images_from_z
                )

        return fake_images, loss_dict
