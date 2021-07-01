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


class TrainStepErrorDistribution(object):
    """Class that contains methods concerning train steps.
    """
    def __init__(self):
        """Instantiate instance of `TrainStepErrorDistribution`.
        """
        pass

    def distributed_eager_error_distribution_train_step(self, features):
        """Perform one distributed, eager error_distribution train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss tensor.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_error_distribution, kwargs={"features": features}
        )

        loss = self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

        return loss

    def non_distributed_eager_error_distribution_train_step(self, features):
        """Perform one non-distributed, eager error_distribution train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss tensor.
        """
        return self.train_error_distribution(features=features)

    @tf.function
    def distributed_graph_error_distribution_train_step(self, features):
        """Perform one distributed, graph error_distribution train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss tensor.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_error_distribution, kwargs={"features": features}
        )

        loss = self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

        return loss

    @tf.function
    def non_distributed_graph_error_distribution_train_step(self, features):
        """Perform one non-distributed, graph error_distribution train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss tensor.
        """
        return self.train_error_distribution(features=features)

    def get_train_step_functions_error_distribution(self):
        """Gets network model train step functions for strategy and mode.
        """
        if self.strategy:
            if self.params["training"]["use_graph_mode"]:
                self.error_distribution_train_step_fn = (
                    self.distributed_graph_error_distribution_train_step
                )
            else:
                self.error_distribution_train_step_fn = (
                    self.distributed_eager_error_distribution_train_step
                )
        else:
            if self.params["training"]["use_graph_mode"]:
                self.error_distribution_train_step_fn = (
                    self.non_distributed_graph_error_distribution_train_step
                )
            else:
                self.error_distribution_train_step_fn = (
                    self.non_distributed_eager_error_distribution_train_step
                )
