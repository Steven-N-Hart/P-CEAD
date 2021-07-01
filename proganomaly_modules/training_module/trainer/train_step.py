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

from . import train_step_reconstruction
from . import train_step_error_distribution
from . import train_step_dynamic_threshold


class TrainStep(
    train_step_reconstruction.TrainStepReconstruction,
    train_step_error_distribution.TrainStepErrorDistribution,
    train_step_dynamic_threshold.TrainStepDynamicThreshold
):
    """Class that contains methods concerning train steps.
    """
    def __init__(self):
        """Instantiate instance of `TrainStep`.
        """
        pass

    @tf.function
    def increment_step_vars(self):
        """Increments step variables.
        """
        self.global_step_var.assign_add(
            delta=tf.ones(shape=(), dtype=tf.int64)
        )

        self.growth_step_var.assign_add(
            delta=tf.ones(shape=(), dtype=tf.int64)
        )

        self.epoch_step_var.assign_add(
            delta=tf.ones(shape=(), dtype=tf.int64)
        )

    def network_model_training_steps_post_reconstruction(
        self, train_step_fn, train_dataset_iter, training_phase
    ):
        """Trains dynamic threshold for so many steps given a set of features.

        Args:
            train_step_fn: unbound function, trains the given network model
                given a set of features.
            train_dataset_iter: iterator, training dataset iterator.
            training_phase: str, which post-reconstruction training phase
                we're currently training: error_distribution or
                dynamic_threshold.
        """
        # Train model on batch of features and get loss.
        if self.params["training"][training_phase]["label_feature_name"]:
            features, labels = next(train_dataset_iter)
        else:
            features = next(train_dataset_iter)

        # Train for a step and get losses.
        _ = train_step_fn(features=features)

        # Checkpoint model every save_checkpoints_steps steps.
        checkpoint_saved = self.checkpoint_manager.save(
            checkpoint_number=self.epoch_step_var, check_interval=True
        )

        # Write logs to disk if checkpoint was saved.
        if checkpoint_saved:
            print("Checkpoint saved at {}".format(checkpoint_saved))

        # Increment steps.
        self.increment_step_vars()
