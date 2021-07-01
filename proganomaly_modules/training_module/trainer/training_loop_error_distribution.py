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


class TrainingLoopErrorDistribution(object):
    """Class that contains methods for error distribution training loop.
    """
    def __init__(self):
        """Instantiate instance of `TrainingLoopErrorDistribution`.
        """
        pass

    @tf.function
    def assign_inverse_covariance_matrix(self, covariance_matrix):
        """Assigns inverse covariance matrix tf.Variable.

        Args:
            covariance_matrix: tensor, rank 2 of shape (num_cols, num_cols)
                containing covariance matrix.
        """
        self.error_distribution_sigma_linv.assign(
            value=tf.linalg.cholesky(
                input=tf.linalg.inv(
                    input=covariance_matrix
                )
            )
        )

    def epoch_steps_loop_error_distribution(self, steps_per_epoch):
        """Loops over steps within current epoch for error distribution.

        Args:
            steps_per_epoch: int, number of train steps to take per epoch.
        """
        self.reset_epoch_step_var()

        self.create_checkpoint_manager_error_distribution()

        while self.epoch_step_var.numpy() < steps_per_epoch:
            self.network_model_training_steps_post_reconstruction(
                train_step_fn=self.error_distribution_train_step_fn,
                train_dataset_iter=(
                    self.train_dataset_error_distribution
                ),
                training_phase="error_distribution"
            )

        self.assign_inverse_covariance_matrix(
            covariance_matrix=(
                self.network_objects["error_distribution"].covariance_matrix
            )
        )

    def training_loop_error_distribution(self):
        """Loops through training dataset to train error distribution model.
        """
        # Get correct train function based on parameters.
        self.get_train_step_functions_error_distribution()

        # Assign current training phase.
        self.assign_training_phase(
            training_phase=tf.constant(value=1, dtype=tf.int64)
        )

        # Loop over steps for one epoch for training job.
        dataset_len = (
            self.params["training"]["error_distribution"]["train_dataset_length"]
        )
        global_batch_size = self.global_batch_size_error_distribution
        self.epoch_steps_loop_error_distribution(
            steps_per_epoch=dataset_len // global_batch_size
        )
