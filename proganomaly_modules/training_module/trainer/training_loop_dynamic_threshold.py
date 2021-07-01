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


class TrainingLoopDynamicThreshold(object):
    """Class that contains methods for dynamic threshold training loop.
    """
    def __init__(self):
        """Instantiate instance of `TrainingLoopDynamicThreshold`.
        """
        pass

    def calculate_dynamic_threshold_supervised(self):
        """Calculates supervised dynamic threshold maximizing f-beta score.

        Returns:
            Scalar tensor dynamic threshold.
        """
        # TODO: Fill this in later.
        return self.dynamic_threshold

    def calculate_dynamic_threshold_unsupervised(self):
        """Calculates unsupervised dynamic threshold using z-test.

        Returns:
            Scalar tensor dynamic threshold.
        """
        dynamic_dict = self.params["training"]["dynamic_threshold"]
        dynamic_threshold_object = self.network_objects["dynamic_threshold"]
        mu = dynamic_threshold_object.col_means_vector[0]
        sigma = dynamic_threshold_object.covariance_matrix[0, 0]
        max_stddevs = dynamic_dict["unsupervised"]["max_mahalanobis_stddevs"]

        return mu + max_stddevs * sigma

    @tf.function
    def assign_dynamic_threshold(self, dynamic_threshold):
        """Assigns dynamic threshold tf.Variable.

        Args:
            dynamic_threshold: tensor, rank 0 scalar that contains value of
                dynamic Mahalanobis distance threshold.
        """
        self.dynamic_threshold.assign(dynamic_threshold)

    def epoch_steps_loop_dynamic_threshold(self, steps_per_epoch):
        """Loops over steps within current epoch for error distribution.

        Args:
            steps_per_epoch: int, number of train steps to take per epoch.
        """
        self.reset_epoch_step_var()

        self.create_checkpoint_manager_dynamic_threshold()

        while self.epoch_step_var.numpy() < steps_per_epoch:
            self.network_model_training_steps_post_reconstruction(
                train_step_fn=self.dynamic_threshold_train_step_fn,
                train_dataset_iter=(
                    self.train_dataset_dynamic_threshold
                ),
                training_phase="dynamic_threshold"
            )

        if self.params["training"]["dynamic_threshold"]["use_supervised"]:
            dynamic_threshold=(
                self.calculate_dynamic_threshold_supervised()
            )
        else:
            dynamic_threshold=(
                self.calculate_dynamic_threshold_unsupervised()
            )
        self.assign_dynamic_threshold(dynamic_threshold)

    def training_loop_dynamic_threshold(self):
        """Loops through training dataset to train error distribution model.
        """
        # Get correct train function based on parameters.
        self.get_train_step_functions_dynamic_threshold()

        # Assign current training phase.
        self.assign_training_phase(
            training_phase=tf.constant(value=2, dtype=tf.int64)
        )

        # Loop over steps for one epoch for training job.
        dataset_len = (
            self.params["training"]["dynamic_threshold"]["train_dataset_length"]
        )
        global_batch_size = self.global_batch_size_dynamic_threshold
        self.epoch_steps_loop_dynamic_threshold(
            steps_per_epoch=dataset_len // global_batch_size
        )
