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

from . import training_inputs


class Datasets(object):
    """Class used for getting datasets.
    """
    def __init__(self):
        """Instantiate instance of `Datasets`.
        """
        pass

    def get_train_eval_dataset(
        self,
        num_replicas,
        batch_size,
        block_idx,
        training,
        params,
        file_pattern=None,
        eval_steps=None
    ):
        """Gets train or eval dataset.

        Args:
            num_replicas: int, number of device replicas.
            batch_size: int, number of elements to extract from dataset.
            block_idx: int, resolution block index.
            training: bool, whether training or not.
            params: dict, user passed parameters.
            file_pattern: str, the file pattern if using TF Records.
            eval_steps: int, if not training, this is the number of batches to
                limit the eval dataset.

        Returns:
            A tf.data.Dataset.
        """
        if params["dataset"] == "mnist":
            dataset = training_inputs.mnist_dataset(
                batch_size=batch_size * num_replicas,
                block_idx=block_idx,
                params=params,
                training=training
            )()
        elif (
            params["dataset"] == "cifar10" or
            params["dataset"] == "cifar10_car"
        ):
            dataset = training_inputs.cifar10_dataset(
                batch_size=batch_size * num_replicas,
                block_idx=block_idx,
                params=params,
                training=training
            )()
        elif params["dataset"] == "tf_record":
            dataset = training_inputs.read_tf_record_dataset(
                file_pattern=file_pattern,
                batch_size=batch_size * num_replicas,
                block_idx=block_idx,
                params=params,
                training=training
            )()

        if not training and eval_steps:
            dataset = dataset.take(count=eval_steps)

        return dataset

    def get_reconstruction_datasets(self, num_blocks):
        """Gets reconstruction datasets for training and eval.

        Args:
            num_blocks: int, number of image resolution growth blocks.
        """
        params = self.params["training"]["reconstruction"]
        params.update(
            {
                "projection_dims": self.params["generator"]["projection_dims"]
            }
        )
        for block_idx in range(num_blocks):
            train_dataset = self.get_train_eval_dataset(
                num_replicas=(
                    self.strategy.num_replicas_in_sync
                    if self.params["training"]["distribution_strategy"]
                    else 1
                ),
                batch_size=params["train_batch_size_schedule"][block_idx],
                block_idx=block_idx,
                training=True,
                params=params,
                file_pattern=params["train_file_patterns"][block_idx]
            )
            self.train_datasets_reconstruction.append(train_dataset)

            eval_dataset = self.get_train_eval_dataset(
                num_replicas=(
                    self.strategy.num_replicas_in_sync
                    if self.params["training"]["distribution_strategy"]
                    else 1
                ),
                batch_size=params["eval_batch_size_schedule"][block_idx],
                block_idx=block_idx,
                training=False,
                params=params,
                file_pattern=params["eval_file_patterns"][block_idx],
                eval_steps=params["eval_steps"]
            )
            self.eval_datasets_reconstruction.append(eval_dataset)

    def get_error_distribution_datasets(self, num_blocks):
        """Gets error distribution datasets for training.

        Args:
            num_blocks: int, number of image resolution growth blocks.
        """
        params = self.params["training"]["error_distribution"]
        params.update(
            {
                "projection_dims": self.params["generator"]["projection_dims"]
            }
        )
        train_dataset = self.get_train_eval_dataset(
            num_replicas=(
                self.strategy.num_replicas_in_sync
                if self.params["training"]["distribution_strategy"]
                else 1
            ),
            batch_size=params["train_batch_size"],
            block_idx=num_blocks - 1,
            training=False,
            params=params,
            file_pattern=params["train_file_pattern"],
            eval_steps=None
        )
        self.train_dataset_error_distribution = train_dataset

    def get_dynamic_threshold_datasets(self, num_blocks):
        """Gets dynamic threshold datasets for training.

        Args:
            num_blocks: int, number of image resolution growth blocks.
        """
        params = self.params["training"]["dynamic_threshold"]
        params.update(
            {
                "projection_dims": self.params["generator"]["projection_dims"]
            }
        )
        train_dataset = self.get_train_eval_dataset(
            num_replicas=(
                self.strategy.num_replicas_in_sync
                if self.params["training"]["distribution_strategy"]
                else 1
            ),
            batch_size=params["train_batch_size"],
            block_idx=num_blocks - 1,
            training=False,
            params=params,
            file_pattern=params["train_file_pattern"],
            eval_steps=None
        )
        self.train_dataset_dynamic_threshold = train_dataset

    def get_all_datasets(self):
        """Gets all datasets for training, eval, etc.
        """
        num_blocks = (self.num_growths + 1) // 2
        # Reconstruction.
        self.get_reconstruction_datasets(num_blocks)

        # Error distribution.
        if self.params["training"]["train_error_distribution"]:
            self.get_error_distribution_datasets(num_blocks)

        # Dynamic threshold.
        if self.params["training"]["train_dynamic_threshold"]:
            self.get_dynamic_threshold_datasets(num_blocks)

        if self.params["training"]["distribution_strategy"]:
            with self.strategy.scope():
                # Create distributed datasets.
                # Reconstruction.
                self.train_datasets_reconstruction = [
                    self.strategy.experimental_distribute_dataset(dataset=x)
                    for x in self.train_datasets_reconstruction
                ]

                self.eval_datasets_reconstruction = [
                    self.strategy.experimental_distribute_dataset(dataset=x)
                    for x in self.eval_datasets_reconstruction
                ]

                # Error distribution.
                if self.params["training"]["train_error_distribution"]:
                    self.train_dataset_error_distribution = (
                        self.strategy.experimental_distribute_dataset(
                            dataset=self.train_dataset_error_distribution
                        )
                    )

                # Dynamic threshold.
                if self.params["training"]["train_dynamic_threshold"]:
                    self.train_dataset_dynamic_threshold = (
                        self.strategy.experimental_distribute_dataset(
                            dataset=self.train_dataset_dynamic_threshold
                        )
                    )
