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


class BatchCalculateDistributionStatistics(object):
    """Class that batch updates distribution statistics.

    Attributes:
        params: dict, user passed parameters.
        num_cols: int, number of columns of rank 2 data matrices.
        seen_example_count: tf.Variable, rank 0 of shape () containing
            the count of the number of examples seen so far.
        col_means_vector: tf.Variable, rank 1 of shape (num_cols,) containing
            column means.
        covariance_matrix: tf.Variable, rank 2 of shape (num_cols, num_cols)
            containing covariance matrix.
    """
    def __init__(self, params, num_cols):
        """Initializes `BatchCalculateDistributionStatistics` class instance.

        Args:
            params: dict, user passed parameters.
            num_cols: int, number of columns of rank 2 data matrices.
        """
        self.params = params
        self.num_cols = num_cols

        self.seen_example_count = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.int64), trainable=False
        )

        self.col_means_vector = tf.Variable(
            initial_value=tf.zeros(
                shape=(self.num_cols,), dtype=tf.float32
            ),
            trainable=False
        )

        self.covariance_matrix = tf.Variable(
            initial_value=tf.zeros(
                shape=(self.num_cols, self.num_cols),
                dtype=tf.float32
            ),
            trainable=False
        )

    @tf.function
    def assign_seen_example_count(self, seen_example_count):
        """Assigns seen example count tf.Variable.

        Args:
            seen_example_count: tensor, rank 0 of shape () containing
            the count of the number of examples seen so far.
        """
        self.seen_example_count.assign(value=seen_example_count)

    @tf.function
    def assign_col_means_vector(self, col_means_vector):
        """Assigns column means vector tf.Variable.

        Args:
            col_means_vector: tensor, rank 1 of shape (num_cols,) containing
            column means.
        """
        self.col_means_vector.assign(value=col_means_vector)

    @tf.function
    def assign_covariance_matrix(self, covariance_matrix):
        """Assigns covariance matrix tf.Variable.

        Args:
            covariance_matrix: tensor, rank 2 of shape (num_cols, num_cols)
            containing covariance matrix.
        """
        self.covariance_matrix.assign(value=covariance_matrix)

    def update_example_count(self, count_a, count_b):
        """Updates the running number of examples processed.

        Given previous running total and current batch size, return new
        running total.

        Args:
            count_a: tensor, tf.int64 rank 0 tensor of previous running total
                of examples.
            count_b: tensor, tf.int64 rank 0 tensor of current batch size.

        Returns:
            A tf.int64 rank 0 tensor of new running total of examples.
        """
        return count_a + count_b

    def update_mean_incremental(self, count_a, mean_a, value_b):
        """Updates the running mean vector incrementally.

        Given previous running total, running column means, and single
            example's column values, return new running column means.

        Args:
            count_a: tensor, tf.int64 rank 0 tensor of previous running total
                of examples.
            mean_a: tensor, tf.float32 rank 1 tensor of previous running column
                means.
            value_b: tensor, tf.float32 rank 1 tensor of single example's
                column values.

        Returns:
            A tf.float32 rank 1 tensor of new running column means.
        """
        umean_a = mean_a * tf.cast(x=count_a, dtype=tf.float32)
        mean_ab_num = umean_a + tf.squeeze(input=value_b, axis=0)
        mean_ab = mean_ab_num / tf.cast(x=count_a + 1, dtype=tf.float32)

        return mean_ab

    def update_covariance_incremental(
        self, count_a, mean_a, cov_a, value_b, mean_ab, use_sample_covariance
    ):
        """Updates the running covariance matrix incrementally.

        Given previous running total, running column means, running covariance
        matrix, single example's column values, new running column means, and
        whether to use sample covariance or not, return new running covariance
        matrix.

        Args:
            count_a: tensor, tf.int64 rank 0 tensor of previous running total
                of examples.
            mean_a: tensor, tf.float32 rank 1 tensor of previous running column
                means.
            cov_a: tensor, tf.float32 rank 2 tensor of previous running
                covariance matrix.
            value_b: tensor, tf.float32 rank 1 tensor of single example's
                column values.
            mean_ab: tensor, tf.float32 rank 1 tensor of new running column
                means.
            use_sample_covariance: bool, flag on whether sample or population
                covariance is used.

        Returns:
            A tf.float32 rank 2 tensor of new covariance matrix.
        """
        mean_diff = tf.matmul(
                a=value_b - mean_a, b=value_b - mean_ab, transpose_a=True
        )

        if use_sample_covariance:
            ucov_a = cov_a * tf.cast(x=count_a - 1, dtype=tf.float32)
            cov_ab_denominator = tf.cast(x=count_a, dtype=tf.float32)
        else:
            ucov_a = cov_a * tf.cast(x=count_a, dtype=tf.float32)
            cov_ab_denominator = tf.cast(x=count_a + 1, dtype=tf.float32)
        cov_ab_numerator = ucov_a + mean_diff
        cov_ab = cov_ab_numerator / cov_ab_denominator

        return cov_ab

    def singleton_batch_update(
        self,
        X,
        running_count,
        running_mean,
        running_covariance,
        use_sample_covariance
    ):
        """Updates running tensors incrementally when batch_size equals 1.

        Given the the data vector X, the tensor tracking running example
        counts, the tensor tracking running column means, and the tensor
        tracking running covariance matrix, returns updated running example
        count tensor, column means tensor, and covariance matrix tensor.

        Args:
            X: tensor, tf.float32 rank 2 tensor of input data.
            running_count: tensor, tf.int64 rank 0 tensor tracking running
                example counts.
            running_mean: tensor, tf.float32 rank 1 tensor tracking running
                column means.
            running_covariance: tensor, tf.float32 rank 2 tensor tracking
                running covariance matrix.
            use_sample_covariance: bool, flag on whether sample or population
                covariance is used.

        Returns:
            Updated updated running example count tensor, column means tensor,
                and covariance matrix tensor.
        """
        # shape = (num_cols, num_cols)
        if running_count == 0:
            # Would produce NaNs, so rollover example for next iteration.
            self.rollover_singleton_example = X

            # Update count though so that we don't end up in this block again.
            count = self.update_example_count(
                count_a=running_count, count_b=1
            )

            # No need to update mean or covariance this iteration
            mean = running_mean
            covariance = running_covariance
        elif running_count == 1:
            # Batch update since we're combining previous & current batches.
            count, mean, covariance = self.non_singleton_batch_update(
                batch_size=2,
                X=tf.concat(
                    values=[self.rollover_singleton_example, X], axis=0
                ),
                running_count=0,
                running_mean=running_mean,
                running_covariance=running_covariance,
                use_sample_covariance=use_sample_covariance
            )
        else:
            # Calculate new combined mean for incremental covariance matrix.
            # shape = (num_cols,)
            mean = self.update_mean_incremental(
                count_a=running_count, mean_a=running_mean, value_b=X
            )

            # Update running tensors from single example
            # shape = ()
            count = self.update_example_count(
                count_a=running_count, count_b=1
            )

            # shape = (num_cols, num_cols)
            covariance = self.update_covariance_incremental(
                count_a=running_count,
                mean_a=running_mean,
                cov_a=running_covariance,
                value_b=X,
                mean_ab=mean,
                use_sample_covariance=use_sample_covariance
            )

        return count, mean, covariance

    def update_mean_batch(self, count_a, mean_a, count_b, mean_b):
        """Updates the running mean vector with a batch of data.

        Given previous running example count, running column means, current
        batch size, and batch's column means, return new running column means.

        Args:
            count_a: tensor, tf.int64 rank 0 tensor of previous running total
                of examples.
            mean_a: tensor, tf.float32 rank 1 tensor of previous running column
                means.
            count_b: tensor, tf.int64 rank 0 tensor of current batch size.
            mean_b: tensor, tf.float32 rank 1 tensor of batch's column means.

        Returns:
            A tf.float32 rank 1 tensor of new running column means.
        """
        sum_a = mean_a * tf.cast(x=count_a, dtype=tf.float32)
        sum_b = mean_b * tf.cast(x=count_b, dtype=tf.float32)
        mean_ab_denominator = tf.cast(x=count_a + count_b, dtype=tf.float32)
        mean_ab = (sum_a + sum_b) / mean_ab_denominator

        return mean_ab

    def update_covariance_batch(
        self,
        count_a,
        mean_a,
        cov_a,
        count_b,
        mean_b,
        cov_b,
        use_sample_covariance
    ):
        """Updates the running covariance matrix with batch of data.

        Given previous running example count, column means, and
        covariance matrix, current batch size, column means, and covariance
        matrix, and whether to use sample covariance or not, return new running
        covariance matrix.

        Args:
            count_a: tensor, tf.int64 rank 0 tensor of previous running total
                of examples.
            mean_a: tensor, tf.float32 rank 1 tensor of previous running column
                means.
            cov_a: tensor, tf.float32 rank 2 tensor of previous running
                covariance matrix.
            count_b: tensor, tf.int64 rank 0 tensor of current batch size.
            mean_b: tensor, tf.float32 rank 1 tensor of batch's column means.
            cov_b: tensor, tf.float32 rank 2 tensor of batch's covariance
                matrix.
            use_sample_covariance: bool, flag on whether sample or population
                covariance is used.

        Returns:
            A tf.float32 rank 2 tensor of new running covariance matrix.
        """
        mean_diff = tf.expand_dims(input=mean_a - mean_b, axis=0)

        if use_sample_covariance:
            ucov_a = cov_a * tf.cast(x=count_a - 1, dtype=tf.float32)
            ucov_b = cov_b * tf.cast(x=count_b - 1, dtype=tf.float32)
            den = tf.cast(x=count_a + count_b - 1, dtype=tf.float32)
        else:
            ucov_a = cov_a * tf.cast(x=count_a, dtype=tf.float32)
            ucov_b = cov_b * tf.cast(x=count_b, dtype=tf.float32)
            den = tf.cast(x=count_a + count_b, dtype=tf.float32)

        mean_diff = tf.matmul(a=mean_diff, b=mean_diff, transpose_a=True)
        mean_scaling_num = tf.cast(x=count_a * count_b, dtype=tf.float32)
        mean_scaling_den = tf.cast(x=count_a + count_b, dtype=tf.float32)
        mean_scaling = mean_scaling_num / mean_scaling_den
        cov_ab = (ucov_a + ucov_b + mean_diff * mean_scaling) / den

        return cov_ab

    def non_singleton_batch_update(
        self,
        batch_size,
        X,
        running_count,
        running_mean,
        running_covariance,
        use_sample_covariance
    ):
        """Updates running tensors when batch_size does NOT equal 1.

        Given the current batch size, the data matrix X, the tensor tracking
        running example counts, the tensor tracking running column means, and
        the tensor tracking running covariance matrix, returns updated running
        example count tensor, column means tensor, and covariance matrix
        tensor.

        Args:
            batch_size: int, number of examples in current batch (could be
                partial).
            X: tensor, tf.float32 rank 2 tensor of input data.
            running_count: tensor, tf.int64 rank 0 tensor tracking running
                example counts.
            running_mean: tensor, tf.float32 rank 1 tensor tracking running
                column means.
            running_covariance: tensor, tf.float32 rank 2 tensor tracking
                running covariance matrix.
            use_sample_covariance: bool, flag on whether sample or population
                covariance is used.

        Returns:
            Updated updated running example count tensor, column means tensor,
                and covariance matrix tensor.
        """
        # shape = (num_cols,)
        X_mean = tf.reduce_mean(input_tensor=X, axis=0)

        # shape = (batch_size, num_cols)
        X_centered = X - X_mean

        # shape = (num_cols, num_cols)
        X_cov = tf.matmul(
                a=X_centered,
                b=X_centered,
                transpose_a=True
        )
        X_cov /= tf.cast(x=batch_size - 1, dtype=tf.float32)

        # Update running tensors from batch statistics.
        # shape = ()
        count = self.update_example_count(
            count_a=running_count, count_b=batch_size
        )

        # shape = (num_cols,)
        mean = self.update_mean_batch(
            count_a=running_count,
            mean_a=running_mean,
            count_b=batch_size,
            mean_b=X_mean
        )

        # shape = (num_cols, num_cols)
        covariance = self.update_covariance_batch(
            count_a=running_count,
            mean_a=running_mean,
            cov_a=running_covariance,
            count_b=batch_size,
            mean_b=X_mean,
            cov_b=X_cov,
            use_sample_covariance=use_sample_covariance
        )

        return count, mean, covariance

    def batch_calculate_data_stats(self, data):
        """Calculates statistics of data.

        Args:
            data: tensor, rank 2 tensor of shape
                (current_batch_size, num_cols) containing batch of input data.
        """
        current_batch_size = data.shape[0]

        if current_batch_size == 1:
            (seen_example_count,
             col_means_vector,
             covariance_matrix) = self.singleton_batch_update(
                X=data,
                running_count=self.seen_example_count,
                running_mean=self.col_means_vector,
                running_covariance=self.covariance_matrix,
                use_sample_covariance=self.params["use_sample_covariance"]
            )
        else:
            (seen_example_count,
             col_means_vector,
             covariance_matrix) = self.non_singleton_batch_update(
                batch_size=current_batch_size,
                X=data,
                running_count=self.seen_example_count,
                running_mean=self.col_means_vector,
                running_covariance=self.covariance_matrix,
                use_sample_covariance=self.params["use_sample_covariance"]
            )

        self.assign_seen_example_count(seen_example_count=seen_example_count)
        self.assign_col_means_vector(col_means_vector=col_means_vector)
        self.assign_covariance_matrix(covariance_matrix=covariance_matrix)
