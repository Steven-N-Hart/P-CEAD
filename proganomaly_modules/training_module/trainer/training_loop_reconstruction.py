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


class TrainingLoopReconstruction(object):
    """Class that contains methods for reconstruction training loop.
    """
    def __init__(self):
        """Instantiate instance of `TrainingLoopReconstruction`.
        """
        pass

    @tf.function
    def assign_growth_idx_var(self, growth_idx):
        """Assigns growth index variable with current index.

        Args:
            growth_idx: tensor, current growth index model has progressed to.
        """
        self.growth_idx_var.assign(value=growth_idx)

    @tf.function
    def reset_growth_step_var(self):
        """Resets growth step variable.
        """
        self.growth_step_var.assign(
            value=tf.zeros(shape=(), dtype=tf.int64)
        )

    @tf.function
    def assign_epoch_idx_var(self, epoch_idx):
        """Assigns epoch index variable with current index.

        Args:
            epoch_idx: tensor, current epoch of growth model is in.
        """
        self.epoch_idx_var.assign(value=epoch_idx)

    def epoch_steps_loop_reconstruction(self, steps_per_epoch):
        """Loops over steps within current epoch for reconstruction.

        Args:
            steps_per_epoch: int, number of train steps to take per epoch.

        Returns:
            Bool that says whether the current growth phase is complete.
        """
        num_steps_until_growth = (
            self.num_steps_until_growth_schedule[self.block_idx]
        )

        growth_phase_complete = (
            self.growth_step_var % num_steps_until_growth == 0
        )

        while self.epoch_step_var.numpy() < steps_per_epoch:
            # Train discriminator.
            if (
                self.params["discriminator"]["create"] and
                self.params["discriminator"]["train"]
            ):
                (growth_phase_complete,
                 features,
                 labels) = self.network_model_training_steps_reconstruction(
                    train_step_fn=self.discriminator_train_step_fn,
                    train_steps=self.params["discriminator"]["train_steps"],
                    train_dataset_iter=(
                        self.train_datasets_reconstruction[self.block_idx]
                    ),
                    features=None,
                    labels=None
                )

                if self.restart_training or growth_phase_complete:
                    # Early return if needed.
                    return True

            # Train generator/encoder.
            if (
                self.params["generator"]["train"] or
                (
                    self.params["encoder"]["create"] and
                    self.params["encoder"]["train"]
                )
            ):
                # Whether we'll need to extract more images this phase or not.
                generator_encoder_phase_needs_real_images = (
                    self.generator_encoder_phase_needs_real_images or
                    not self.params["discriminator"]["train"]
                )

                (growth_phase_complete,
                 _,
                 _) = self.network_model_training_steps_reconstruction(
                    train_step_fn=self.generator_train_step_fn,
                    train_steps=self.params["generator"]["train_steps"],
                    train_dataset_iter=(
                        self.train_datasets_reconstruction[self.block_idx]
                        if generator_encoder_phase_needs_real_images
                        else None
                    ),
                    features=(
                        None
                        if generator_encoder_phase_needs_real_images
                        else features
                    ),
                    labels=(
                        None
                        if generator_encoder_phase_needs_real_images
                        else labels
                    )
                )

                if self.restart_training or growth_phase_complete:
                    # Early return if needed.
                    return True

        # After while loop return if there is more training of current growth.
        return growth_phase_complete

    def get_growths_epoch_loop_limits(self):
        """Gets current growth's epoch loop limits.
        """
        recon_dict = self.params["training"]["reconstruction"]
        num_epochs = recon_dict["num_epochs_schedule"][self.block_idx]
        # If epoch_idx was already set by checkpoint.
        if self.epoch_idx_start >= 0:
            if recon_dict["initial_epoch_idx"] >= 0:
                return (
                    self.epoch_idx_start,
                    recon_dict["initial_epoch_idx"] + num_epochs
                )
            return self.epoch_idx_start, num_epochs

        # If user sent job and wants to override growth_idx.
        if (
            recon_dict["initial_epoch_idx"] >= 0 and
            self.growth_idx == recon_dict["initial_growth_idx"]
        ):
            return (
                recon_dict["initial_epoch_idx"],
                recon_dict["initial_epoch_idx"] + num_epochs
            )

        # Otherwise train from beginning of growth.
        return 0, num_epochs

    def epoch_loop(self, steps_per_epoch):
        """Loops over epochs within current growth.

        Args:
            steps_per_epoch: int, number of train steps to take per epoch.
        """
        recon_dict = self.params["training"]["reconstruction"]
        # Get epoch loop's limits for this growth phase.
        (start_epoch_idx,
         end_epoch_idx) = self.get_growths_epoch_loop_limits()

        for self.epoch_idx in range(start_epoch_idx, end_epoch_idx):
            if self.epoch_idx_start == -1:
                self.reset_epoch_step_var()
            else:
                self.epoch_idx_start = -1
            self.assign_epoch_idx_var(
                epoch_idx=tf.convert_to_tensor(
                    value=self.epoch_idx, dtype=tf.int64
                )
            )
            print(
                "\ngrowth_idx = {}, epoch_idx = {}".format(
                    self.growth_idx, self.epoch_idx
                )
            )

            self.previous_timestamp = tf.timestamp()

            if recon_dict["checkpoint_every_epoch"]:
                # Create new checkpoint manager for current growth and epoch.
                self.create_checkpoint_manager_reconstruction(
                    growth_idx=self.growth_idx,
                    epoch_idx=self.epoch_idx
                )

            # Loop over steps within current epoch.
            growth_phase_complete = self.epoch_steps_loop_reconstruction(
                steps_per_epoch
            )

            if self.restart_training or growth_phase_complete:
                # Need to restart or done with this growth phase, return early.
                return
            else:
                if recon_dict["checkpoint_every_epoch"]:
                    # Checkpoint model at end of epoch.
                    checkpoint_saved = self.checkpoint_manager.save(
                        checkpoint_number=self.epoch_step_var,
                        check_interval=False
                    )

                    # Write logs to disk if checkpoint was saved.
                    if checkpoint_saved:
                        print(
                            "Checkpoint saved at {}".format(checkpoint_saved)
                        )
                        if recon_dict["store_loss_logs"]:
                            self.write_loss_logs()
                else:
                    if recon_dict["store_loss_logs"]:
                        self.write_loss_logs()

                if self.params["export"]["export_every_epoch"]:
                    self.export_saved_model_reconstruction()

    def get_growth_loop_limits(self):
        """Gets growth loop's limits.
        """
        recon_dict = self.params["training"]["reconstruction"]
        # If growth_idx was already set by checkpoint.
        if self.growth_idx_start >= 0:
            return self.growth_idx_start, self.num_growths

        # If user sent job and wants to override growth_idx.
        if recon_dict["initial_growth_idx"] >= 0:
            return recon_dict["initial_growth_idx"], self.num_growths

        # Otherwise train from scratch.
        return 0, self.num_growths

    def growth_loop(self):
        """Loops over growths within training job.
        """
        recon_dict = self.params["training"]["reconstruction"]
        start_growth_idx, end_growth_idx = self.get_growth_loop_limits()
        for self.growth_idx in range(start_growth_idx, end_growth_idx):
            if self.growth_idx_start == -1:
                self.reset_growth_step_var()
            else:
                self.growth_idx_start = -1
            self.assign_growth_idx_var(
                growth_idx=tf.convert_to_tensor(
                    value=self.growth_idx, dtype=tf.int64
                )
            )

            self.block_idx = (self.growth_idx + 1) // 2
            if recon_dict["print_training_model_summaries"]:
                print(
                    "\nblock_idx = {}, growth_idx = {}".format(
                        self.block_idx, self.growth_idx
                    )
                )
                print(
                    "\ngenerator_model = {}".format(
                        self.network_objects["generator"].models[self.growth_idx].model(training=True).summary()
                    )
                )
                print(
                    "\nencoder_model = {}".format(
                        self.network_objects["encoder"].models[self.growth_idx].model(training=True).summary()
                    )
                )
                print(
                    "\ndiscriminator_model = {}".format(
                        self.network_objects["discriminator"].models[self.growth_idx].model(training=True).summary()
                    )
                )

            global_batch_size = (
                self.global_batch_size_schedule_reconstruction[self.block_idx]
            )
            steps_per_epoch = (
                recon_dict["train_dataset_length"] // global_batch_size
            )

            # Loop over epochs within current growth.
            self.epoch_loop(steps_per_epoch)

            # If training needs to be restarted, exit back to outer scope.
            if self.restart_training:
                return

            if recon_dict["checkpoint_every_growth_phase"]:
                if not recon_dict["checkpoint_every_epoch"]:
                    # Create checkpoint manager for current growth & epoch.
                    self.create_checkpoint_manager_reconstruction(
                        growth_idx=self.growth_idx,
                        epoch_idx=self.epoch_idx
                    )

                # Checkpoint model at end of growth.
                checkpoint_saved = self.checkpoint_manager.save(
                    checkpoint_number=self.epoch_step_var,
                    check_interval=False
                )

                # Write logs to disk if checkpoint was saved.
                if checkpoint_saved:
                    print("Checkpoint saved at {}".format(checkpoint_saved))
                    if recon_dict["store_loss_logs"]:
                        self.write_loss_logs()
            else:
                if recon_dict["store_loss_logs"]:
                    self.write_loss_logs()

            if self.params["export"]["export_every_growth_phase"]:
                self.export_saved_model_reconstruction()

    def training_loop_reconstruction(self):
        """Loops through training dataset to train reconstruction model.
        """
        # Get correct train function based on parameters.
        self.get_train_step_functions_reconstruction()

        # Assign current training phase.
        self.assign_training_phase(
            training_phase=tf.constant(value=0, dtype=tf.int64)
        )

        # Loop over growths for training job.
        self.growth_loop()

        # If training needs to be restarted, exit back to outer scope.
        if self.restart_training:
            return
