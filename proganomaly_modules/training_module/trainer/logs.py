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

import datetime
import json
import os
import tensorflow as tf


class Logs(object):
    """Class that contains methods concerning logs.
    """
    def __init__(self):
        """Instantiate instance of `Logs`.
        """
        pass

    def log_step_loss(self):
        """Logs step information and loss.
        """
        reconstruction_dict = self.params["training"]["reconstruction"]

        log_step_count_steps = reconstruction_dict["log_step_count_steps"]
        if self.epoch_step_var % log_step_count_steps == 0:
            start_time = self.previous_timestamp
            self.previous_timestamp = tf.timestamp()
            elapsed_time = self.previous_timestamp - start_time
            steps_per_sec = float(log_step_count_steps) / elapsed_time

            full_log = {
                "datetime_utc": str(
                    datetime.datetime.fromtimestamp(self.previous_timestamp)
                ),
                "block_idx": str(self.block_idx),
                "growth_idx": str(self.growth_idx),
                "epoch_idx": str(self.epoch_idx)
            }

            loss_log = {
                "global_step": str(self.global_step_var.numpy()),
                "growth_step": str(self.growth_step_var.numpy()),
                "epoch_step": str(self.epoch_step_var.numpy()),
                "steps/sec": str(steps_per_sec.numpy()),
                "losses": {
                    network: str(loss.numpy())
                    for network, loss in self.losses.items()
                }
            }

            full_log.update(loss_log)

            if reconstruction_dict["store_loss_logs"]:
                if reconstruction_dict["normalized_loss_logs"]:
                    self.loss_logs.append(loss_log)
                else:
                    self.loss_logs.append(full_log)

            print(full_log)

    def write_loss_logs(self):
        """Writes loss logs to disk.
        """
        if self.loss_logs:
            tf.io.write_file(
                filename=os.path.join(
                    self.params["training"]["output_dir"],
                    "loss_logs",
                    "loss_logs_growth_{}_epoch_{}_epoch_step_{}.json".format(
                        self.growth_idx,
                        self.epoch_idx,
                        self.epoch_step_var.numpy()
                    )
                ),
                contents=json.dumps(self.loss_logs)
            )

            # Reset logs.
            self.loss_logs = []
