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

from . import training_loop_reconstruction
from . import training_loop_error_distribution
from . import training_loop_dynamic_threshold


class TrainingLoop(
    training_loop_reconstruction.TrainingLoopReconstruction,
    training_loop_error_distribution.TrainingLoopErrorDistribution,
    training_loop_dynamic_threshold.TrainingLoopDynamicThreshold
):
    """Class that contains methods for training loop.
    """
    def __init__(self):
        """Instantiate instance of `TrainingLoop`.
        """
        pass

    @tf.function
    def assign_training_phase(self, training_phase):
        """Assigns current training_phase.
        """
        self.training_phase.assign(
            value=training_phase
        )

    @tf.function
    def reset_epoch_step_var(self):
        """Resets epoch step variable.
        """
        self.epoch_step_var.assign(
            value=tf.zeros(shape=(), dtype=tf.int64)
        )
