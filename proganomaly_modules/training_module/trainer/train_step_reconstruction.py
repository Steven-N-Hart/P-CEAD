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


class TrainStepReconstruction(object):
    """Class that contains methods concerning train steps.
    """
    def __init__(self):
        """Instantiate instance of `TrainStepReconstruction`.
        """
        pass

    def distributed_eager_discriminator_train_step(
        self, features, growth_idx
    ):
        """Perform one distributed, eager discriminator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_discriminator,
            kwargs={"features": features}
        )

        losses = {
            k: self.strategy.reduce(
                reduce_op=tf.distribute.ReduceOp.SUM,
                value=v,
                axis=None
            )
            for k, v in per_replica_losses.items()
        }

        return losses

    def non_distributed_eager_discriminator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, eager discriminator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        return self.train_discriminator(features=features)

    @tf.function
    def distributed_graph_discriminator_train_step(
        self, features, growth_idx
    ):
        """Perform one distributed, graph discriminator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_discriminator,
            kwargs={"features": features}
        )

        losses = {
            k: self.strategy.reduce(
                reduce_op=tf.distribute.ReduceOp.SUM,
                value=v,
                axis=None
            )
            for k, v in per_replica_losses.items()
        }

        return losses

    @tf.function
    def non_distributed_graph_discriminator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, graph discriminator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        return self.train_discriminator(features=features)

    def distributed_eager_generator_train_step(self, features, growth_idx):
        """Perform one distributed, eager generator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_generator_encoder,
            kwargs={"features": features}
        )

        losses = {
            k: self.strategy.reduce(
                reduce_op=tf.distribute.ReduceOp.SUM,
                value=v,
                axis=None
            )
            for k, v in per_replica_losses.items()
        }

        return losses

    def non_distributed_eager_generator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, eager generator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        return self.train_generator_encoder(features=features)

    @tf.function
    def distributed_graph_generator_train_step(self, features, growth_idx):
        """Perform one distributed, graph generator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_generator_encoder,
            kwargs={"features": features}
        )

        losses = {
            k: self.strategy.reduce(
                reduce_op=tf.distribute.ReduceOp.SUM,
                value=v,
                axis=None
            )
            for k, v in per_replica_losses.items()
        }

        return losses

    @tf.function
    def non_distributed_graph_generator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, graph generator train step.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.
                Solely passed due to how tf.function creates graphs.

        Returns:
            Dictionary of scalar losses for each network.
        """
        return self.train_generator_encoder(features=features)

    def get_train_step_functions_reconstruction(self):
        """Gets network model train step functions for strategy and mode.
        """
        if self.strategy:
            if self.params["training"]["use_graph_mode"]:
                self.discriminator_train_step_fn = (
                    self.distributed_graph_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.distributed_graph_generator_train_step
                )
            else:
                self.discriminator_train_step_fn = (
                    self.distributed_eager_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.distributed_eager_generator_train_step
                )
        else:
            if self.params["training"]["use_graph_mode"]:
                self.discriminator_train_step_fn = (
                    self.non_distributed_graph_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.non_distributed_graph_generator_train_step
                )
            else:
                self.discriminator_train_step_fn = (
                    self.non_distributed_eager_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.non_distributed_eager_generator_train_step
                )

    def rollback_to_checkpoint_if_nan_loss(self):
        """Rolls back to earlier checkpoint if there are any NaN losses.
        """
        numpy_losses = {
            network: loss.numpy() for network, loss in self.losses.items()
        }

        for loss in numpy_losses.values():
            # Check for nan.
            if loss != loss:
                print(
                    "There was a NaN loss! losses = {}".format(numpy_losses)
                )

                # Restore latest checkpoint.
                print(
                    "Loading latest checkpoint: {}".format(
                        self.checkpoint_manager.latest_checkpoint
                    )
                )
                status = self.checkpoint.restore(
                    save_path=self.checkpoint_manager.latest_checkpoint
                )

                if self.checkpoint_manager.latest_checkpoint:
                    status.assert_consumed()

                print(
                    "Setting {} = {} & {} = {}".format(
                        "growth_idx_start",
                        self.growth_idx_var.numpy(),
                        "epoch_idx_start",
                        self.epoch_idx_var.numpy()
                    )
                )
                self.growth_idx_start = self.growth_idx_var.numpy()
                self.epoch_idx_start = self.epoch_idx_var.numpy()

                self.restart_training = True

                # Return early since no need to check for more NaNs.
                return

    @tf.function
    def increment_alpha_var(self):
        """Increments alpha variable through range [0., 1.] during transition.
        """
        num_steps_until_growth = (
            self.num_steps_until_growth_schedule[self.block_idx]
        )

        self.alpha_var.assign(
            value=tf.minimum(
                x=tf.divide(
                    x=tf.cast(x=self.growth_step_var + 1, dtype=tf.float32),
                    y=tf.cast(x=num_steps_until_growth, dtype=tf.float32)
                ),
                y=1.0
            )
        )

    def network_model_training_steps_reconstruction(
        self,
        train_step_fn,
        train_steps,
        train_dataset_iter,
        features,
        labels
    ):
        """Trains a network model for so many steps given a set of features.

        Args:
            train_step_fn: unbound function, trains the given network model
                given a set of features.
            train_steps: int, number of steps to train network model.
            train_dataset_iter: iterator, training dataset iterator.
            features: dict, feature tensors from input function.
            labels: tensor, label tensor from input function.

        Returns:
            Bool that indicates if current growth phase complete,
                dictionary of most recent feature tensors, and most recent
                label tensor.
        """
        for _ in range(train_steps):
            if features is None:
                # Train model on batch of features and get loss.
                if self.params["training"]["reconstruction"]["label_feature_name"]:
                    features, labels = next(train_dataset_iter)
                else:
                    features = next(train_dataset_iter)

            # Train for a step and get losses.
            self.losses = train_step_fn(
                features=features, growth_idx=self.growth_idx
            )

            # Check losses aren't NaN.
            self.rollback_to_checkpoint_if_nan_loss()

            # If training needs to be restarted, return so as not to log loss,
            # save checkpoint, or increment steps.
            if self.restart_training:
                return True, features, labels

            # Log step information and loss.
            self.log_step_loss()

            # Checkpoint model every save_checkpoints_steps steps.
            checkpoint_saved = self.checkpoint_manager.save(
                checkpoint_number=self.epoch_step_var, check_interval=True
            )

            # Write logs to disk if checkpoint was saved.
            if checkpoint_saved:
                print("Checkpoint saved at {}".format(checkpoint_saved))
                if self.params["training"]["reconstruction"]["store_loss_logs"]:
                    self.write_loss_logs()

            # Increment steps.
            self.increment_step_vars()

            # If this is a growth transition phase.
            if self.growth_idx % 2 == 1:
                # Increment alpha variable.
                self.increment_alpha_var()

            num_steps_until_growth = (
                self.num_steps_until_growth_schedule[self.block_idx]
            )

            if self.growth_step_var % num_steps_until_growth == 0:
                return True, features, labels
        return False, features, labels
