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

import os
import tensorflow as tf


class Checkpoints(object):
    """Class used for training checkpoints.
    """
    def __init__(self):
        """Instantiate instance of `Checkpoints`.
        """
        pass

    def create_checkpoint_manager_reconstruction(self, growth_idx, epoch_idx):
        """Creates reconstruction checkpoint manager for growth and epoch.

        Args:
            growth_idx: int, current growth index model has progressed to.
            epoch_idx: int, current epoch of growth model is in.
        """
        reconstruction_dict = self.params["training"]["reconstruction"]

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(
                self.params["training"]["output_dir"], "checkpoints"
            ),
            max_to_keep=reconstruction_dict["keep_checkpoint_max"],
            checkpoint_name="growth_{}_epoch_{}_ckpt".format(
                growth_idx, epoch_idx
            ),
            step_counter=self.epoch_step_var,
            checkpoint_interval=reconstruction_dict["save_checkpoints_steps"]
        )

    def create_checkpoint_manager_error_distribution(self):
        """Creates error distribution checkpoint manager.
        """
        error_dist_dict = self.params["training"]["error_distribution"]

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(
                self.params["training"]["output_dir"], "checkpoints"
            ),
            max_to_keep=error_dist_dict["keep_checkpoint_max"],
            checkpoint_name="error_distribution_ckpt",
            step_counter=self.epoch_step_var,
            checkpoint_interval=error_dist_dict["save_checkpoints_steps"]
        )

    def create_checkpoint_manager_dynamic_threshold(self):
        """Creates dynamic threshold checkpoint manager.
        """
        dyn_thresh_dict = self.params["training"]["dynamic_threshold"]

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(
                self.params["training"]["output_dir"], "checkpoints"
            ),
            max_to_keep=dyn_thresh_dict["keep_checkpoint_max"],
            checkpoint_name="dynamic_threshold_ckpt",
            step_counter=self.epoch_step_var,
            checkpoint_interval=dyn_thresh_dict["save_checkpoints_steps"]
        )

    def create_checkpoint_machinery(self):
        """Creates checkpoint machinery needed to save & restore checkpoints.
        """
        reconstruction_dict = self.params["training"]["reconstruction"]
        error_dict = self.params["training"]["error_distribution"]
        dynamic_dict = self.params["training"]["dynamic_threshold"]

        # Create kwargs dict.
        kwargs = {
            "training_phase": self.training_phase,
            "still_training": self.still_training_var,
            "global_step": self.global_step_var,
            "growth_step": self.growth_step_var,
            "epoch_step": self.epoch_step_var,
            "alpha": self.alpha_var,
            "most_recent_export_growth_idx": self.most_recent_export_growth_idx,
            "most_recent_export_epoch_idx": self.most_recent_export_epoch_idx,
            "growth_idx": self.growth_idx_var,
            "epoch_idx": self.epoch_idx_var,
            "generator_variables": (
                self.unique_trainable_variables["generator"]
            ),
            "generator_optimizer": self.optimizers["generator"]
        }

        if self.params["encoder"]["create"]:
            kwargs.update(
                {
                    "encoder_variables": (
                        self.unique_trainable_variables["encoder"]
                    ),
                    "encoder_optimizer": self.optimizers["encoder"],
                }
            )

        if self.params["discriminator"]["create"]:
            kwargs.update(
                {
                    "discriminator_variables": (
                        self.unique_trainable_variables["discriminator"]
                    ),
                    "discriminator_optimizer": self.optimizers["discriminator"],
                }
            )

        if self.params["training"]["train_error_distribution"]:
            kwargs.update(
                {
                    "error_distribution_seen_example_count": (
                        self.network_objects["error_distribution"].seen_example_count
                    ),
                    "error_distribution_col_means_vector": (
                        self.network_objects["error_distribution"].col_means_vector
                    ),
                    "error_distribution_covariance_matrix": (
                        self.network_objects["error_distribution"].covariance_matrix
                    ),
                    "error_distribution_sigma_linv": (
                        self.error_distribution_sigma_linv
                    )
                }
            )

        if self.params["training"]["train_dynamic_threshold"]:
            kwargs.update(
                {
                    "dynamic_threshold_seen_example_count": (
                        self.network_objects["dynamic_threshold"].seen_example_count
                    ),
                    "dynamic_threshold_col_means_vector": (
                        self.network_objects["dynamic_threshold"].col_means_vector
                    ),
                    "dynamic_threshold_covariance_matrix": (
                        self.network_objects["dynamic_threshold"].covariance_matrix
                    ),
                    "dynamic_threshold": self.dynamic_threshold
                }
            )

        # Create checkpoint instance.
        self.checkpoint = tf.train.Checkpoint(**kwargs)

        # Get latest checkpoint.
        latest_checkpoint = tf.train.latest_checkpoint(
            checkpoint_dir=os.path.join(
                self.params["training"]["output_dir"], "checkpoints"
            )
        )

        print("Latest checkpoint = {}".format(latest_checkpoint))

        # Determine which type of checkpoint the latest is, if it exists.
        if latest_checkpoint:
            checkpoint_prefix = latest_checkpoint.split("/")[-1].split("_")[0]

            # Create initial checkpoint manager.
            if checkpoint_prefix == "growth":
                self.create_checkpoint_manager_reconstruction(
                    growth_idx=reconstruction_dict["checkpoint_growth_idx"],
                    epoch_idx=reconstruction_dict["checkpoint_epoch_idx"]
                )
            elif checkpoint_prefix == "error":
                self.create_checkpoint_manager_error_distribution()
            elif checkpoint_prefix == "dynamic":
                self.create_checkpoint_manager_dynamic_threshold()

            # Restore any prior checkpoints.
            status = self.checkpoint.restore(
                save_path=self.checkpoint_manager.latest_checkpoint
            )

            if self.checkpoint_manager.latest_checkpoint:
                status.assert_consumed()

        if any(
            [
                reconstruction_dict["checkpoint_save_path"],
                error_dict["checkpoint_save_path"],
                dynamic_dict["checkpoint_save_path"]
            ]
        ):
            # If there was at some point a user chosen checkpoint path.
            if self.still_training_var.numpy():
                # If latest checkpoint WAS in a training state,
                # i.e. this training job was restarted by the service.
                print("No need to load checkpoint at save path.")
                if checkpoint_prefix == "growth":
                    self.create_checkpoint_manager_reconstruction(
                        self.growth_idx, self.epoch_idx
                    )
                elif checkpoint_prefix == "error":
                    self.create_checkpoint_manager_error_distribution()
                elif checkpoint_prefix == "dynamic":
                    self.create_checkpoint_manager_dynamic_threshold()
            else:
                # If latest checkpoint was NOT in a training state,
                # i.e. this training job was submitted by user.
                if reconstruction_dict["checkpoint_save_path"]:
                    save_path = reconstruction_dict["checkpoint_save_path"]
                elif error_dict["checkpoint_save_path"]:
                    save_path = error_dict["checkpoint_save_path"]
                elif dynamic_dict["checkpoint_save_path"]:
                    save_path = dynamic_dict["checkpoint_save_path"]

                print(
                    "Loading checkpoint at save path = {}".format(save_path)
                )
                status = self.checkpoint.restore(save_path=save_path)
                status.assert_consumed()

        print(
            "still_training_var = {}, training_phase = {}".format(
                self.still_training_var.numpy(), self.training_phase.numpy()
            )
        )

        # Reinitialize class state.
        if self.still_training_var.numpy():
            if self.training_phase.numpy() == 2:
                if self.params["training"]["train_dynamic_threshold"]:
                    self.training_phase_schedule.append(1)
            elif self.training_phase.numpy() == 1:
                if self.params["training"]["train_dynamic_threshold"]:
                    self.training_phase_schedule.append(1)
                if self.params["training"]["train_error_distribution"]:
                    self.training_phase_schedule.append(0)
            elif self.training_phase.numpy() == 0:
                if self.params["training"]["train_dynamic_threshold"]:
                    self.training_phase_schedule.append(1)
                if self.params["training"]["train_error_distribution"]:
                    self.training_phase_schedule.append(0)
                if self.params["training"]["train_reconstruction"]:
                    self.training_phase_schedule.append(-1)

                # If latest checkpoint WAS in a training state,
                # i.e. this training job was restarted by the service.
                print(
                    "Setting growth_idx_start = {} & epoch_idx_start = {}".format(
                        self.growth_idx_var.numpy(), self.epoch_idx_var.numpy()
                    )
                )
                self.growth_idx_start = self.growth_idx_var.numpy()
                self.epoch_idx_start = self.epoch_idx_var.numpy()
        else:
            if self.params["training"]["train_dynamic_threshold"]:
                self.training_phase_schedule.append(1)
            if self.params["training"]["train_error_distribution"]:
                self.training_phase_schedule.append(0)
            if self.params["training"]["train_reconstruction"]:
                self.training_phase_schedule.append(-1)
