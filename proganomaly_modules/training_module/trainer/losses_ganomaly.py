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

from . import image_masks


class LossesGanomaly(image_masks.ImageMasks):
    """Class used for both training & evaluation losses for GANomaly paper.
    """
    def __init__(self):
        """Instantiate instance of `LossesGanomaly`.
        """
        pass

    def encoder_loss_phase_ganomaly(
        self, logits_from_x, fake_images_from_x, training
    ):
        """Gets losses for encoder.

        Args:
            logits_from_x: tensor, images of shape
                (batch_size, generator_latent_size).
            fake_images_from_x: tensor, fake images from generator of shape
                (batch_size, image_height, image_width, depth).
            training: bool, if model should be training.

        Returns:
            2-tuple of encoder loss tensor of shape () for fake images from
                real images and dictionary of encoder losses.
        """
        gen_loss_weights = self.params["generator"]["losses"]["GANomaly"]
        enc_loss_weights = self.params["encoder"]["losses"]["GANomaly"]

        encoder_loss_dict = {}

        if any(
            [
                gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"],
                gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"],
                enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"],
                enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"]
            ]
        ):
            # E(G(x)).
            z_hat_from_fake_images_from_x = (
                self.network_objects["encoder"].models[self.growth_idx](
                    inputs=fake_images_from_x, training=training
                )
            )

            if (
                gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"] or
                enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"]
            ):
                encoder_loss_dict["Ge(x)-E(G(x))_L1"] = (
                    self.get_encoder_l1_z_loss(
                        z=logits_from_x, z_hat=z_hat_from_fake_images_from_x
                    )
                )

            if (
                gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"] or
                enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"]
            ):
                encoder_loss_dict["Ge(x)-E(G(x))_L2"] = (
                    self.get_encoder_l2_z_loss(
                        z=logits_from_x, z_hat=z_hat_from_fake_images_from_x
                    )
                )

        return encoder_loss_dict

    def generator_encoder_loss_phase_ganomaly(self, real_images, training):
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
        generator_params = self.params["generator"]
        gen_loss_weights = generator_params["losses"]["GANomaly"]
        enc_loss_weights = self.params["encoder"]["losses"]["GANomaly"]
        dis_loss_weights = self.params["discriminator"]["losses"]["GANomaly"]
        recon_dict = self.params["training"]["reconstruction"]

        # Create dictionary to contain all raw losses.
        loss_dict = {}

        # Generator phase.
        use_masked_images = (
            generator_params["architecture"] == "GANomaly" and
            generator_params["GANomaly"][
                "mask_generator_input_images_percent"] > 0.
        )

        logits_from_x, fake_images_from_x = (
            self.network_objects["generator"].models[self.growth_idx](
                inputs=(
                    self.mask_images(
                        images=real_images, growth_idx=self.growth_idx
                    )
                    if use_masked_images
                    else real_images
                ),
                training=training
            )
        )

        if (
            generator_params["add_uniform_noise_to_fake_images"] and
            training
        ):
            fake_images_from_x += tf.random.uniform(
                shape=tf.shape(input=fake_images_from_x)
            )

        if gen_loss_weights["x_minus_G_of_x_l1_loss_weight"]:
            loss_dict["x-G(x)_L1"] = (
                self.get_encoder_l1_image_loss(
                    images=real_images,
                    encoded_images=fake_images_from_x
                )
            )

        if gen_loss_weights["x_minus_G_of_x_l2_loss_weight"]:
            loss_dict["x-G(x)_L2"] = (
                self.get_encoder_l2_image_loss(
                    images=real_images,
                    encoded_images=fake_images_from_x
                )
            )

        # Encoder phase.
        encoder_loss_dict = dict()
        if self.params["encoder"]["create"]:
            encoder_loss_dict = self.encoder_loss_phase_ganomaly(
                logits_from_x, fake_images_from_x, training
            )

        # Add any encoder losses to loss dict.
        loss_dict.update(encoder_loss_dict)

        # Discriminator phase.
        if self.params["discriminator"]["create"]:
            if any(
                [
                    gen_loss_weights["D_of_G_of_x_loss_weight"],
                    dis_loss_weights["D_of_G_of_x_loss_weight"]
                ]
            ):
                # D(G(x)).
                loss_dict["D(G(x))"] = self.generator_loss_phase(
                    fake_image_type="D(G(x))",
                    fake_images=fake_images_from_x,
                    training=training
                )

        # Combine losses into generator total loss.
        generator_reg_loss = self.get_network_regularization_loss(
            network="generator"
        )
        loss_dict["generator_reg_loss"] = generator_reg_loss
        generator_total_loss = generator_reg_loss

        if self.params["discriminator"]["create"]:
            if gen_loss_weights["D_of_G_of_x_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["D_of_G_of_x_loss_weight"],
                    y=-loss_dict["D(G(x))"]
                )

        if gen_loss_weights["x_minus_G_of_x_l1_loss_weight"]:
            generator_total_loss += tf.multiply(
                x=gen_loss_weights["x_minus_G_of_x_l1_loss_weight"],
                y=loss_dict["x-G(x)_L1"]
            )

        if gen_loss_weights["x_minus_G_of_x_l2_loss_weight"]:
            generator_total_loss += tf.multiply(
                x=gen_loss_weights["x_minus_G_of_x_l2_loss_weight"],
                y=loss_dict["x-G(x)_L2"]
            )

        if self.params["encoder"]["create"]:
            if gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"],
                    y=loss_dict["Ge(x)-E(G(x))_L1"]
                )

            if gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"]:
                generator_total_loss += tf.multiply(
                    x=gen_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"],
                    y=loss_dict["Ge(x)-E(G(x))_L2"]
                )

        loss_dict["generator_total_loss"] = generator_total_loss

        if self.params["encoder"]["create"]:
            # Combine losses into encoder total loss.
            encoder_reg_loss = self.get_network_regularization_loss(
                network="encoder"
            )
            loss_dict["encoder_reg_loss"] = encoder_reg_loss
            encoder_total_loss = encoder_reg_loss

            if enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"],
                    y=loss_dict["Ge(x)-E(G(x))_L1"]
                )

            if enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"]:
                encoder_total_loss += tf.multiply(
                    x=enc_loss_weights["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"],
                    y=loss_dict["Ge(x)-E(G(x))_L2"]
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
            fake_images["G_of_x"] = fake_images_from_x

        return fake_images, loss_dict
