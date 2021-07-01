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
import os
import tensorflow as tf

from . import export_berg
from . import export_ganomaly


class Export(export_berg.ExportBerg, export_ganomaly.ExportGanomaly):
    """Class used for exporting model objects.
    """
    def __init__(self):
        """Instantiate instance of `Export`.
        """
        pass

    @tf.function
    def assign_most_recent_export_idx_vars(self):
        """Assigns most_recent_export_idx variables with current idx values.
        """
        self.most_recent_export_growth_idx.assign(value=self.growth_idx)
        self.most_recent_export_epoch_idx.assign(value=self.epoch_idx)

    def export_saved_model_reconstruction(self):
        """Exports reconstruction SavedModel to output directory.
        """
        growth_idx = self.growth_idx
        epoch_idx = self.epoch_idx
        if (
            growth_idx == self.most_recent_export_growth_idx and
            epoch_idx == self.most_recent_export_epoch_idx
        ):
            print("Skipping export because already done.")
        else:
            # Build export path.
            export_path = os.path.join(
                self.params["training"]["output_dir"],
                "export",
                "{}_growth_{}_epoch_{}".format(
                    datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                    growth_idx,
                    epoch_idx
                )
            )

            # Create serving models.
            if self.params["generator"]["architecture"] == "berg":
                self.create_serving_models_berg()
            elif self.params["generator"]["architecture"] == "GANomaly":
                self.create_serving_models_ganomaly()

            # Signature will be serving_default.
            self.serving_model.save(
                filepath=export_path,
                overwrite=True,
                include_optimizer=True,
                save_format="tf"
            )

            self.assign_most_recent_export_idx_vars()

    def export_saved_model_post_reconstruction(self, training_phase):
        """Exports post-reconstruction SavedModel to output directory.
        """
        # Build export path.
        export_path = os.path.join(
            self.params["training"]["output_dir"],
            "export",
            "{}_{}".format(
                datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                training_phase
            )
        )

        # Create serving models.
        if self.params["generator"]["architecture"] == "berg":
            self.create_serving_models_berg()
        elif self.params["generator"]["architecture"] == "GANomaly":
            self.create_serving_models_ganomaly()

        # Signature will be serving_default.
        self.serving_model.save(
            filepath=export_path,
            overwrite=True,
            include_optimizer=True,
            save_format="tf"
        )

    def training_loop_end_save_model_reconstruction(self):
        """Saving model when reconstruction training loop ends.
        """
        recon_dict = self.params["training"]["reconstruction"]
        if not recon_dict["checkpoint_every_growth_phase"]:
            if not recon_dict["checkpoint_every_epoch"]:
                # Create checkpoint manager for current growth & epoch.
                self.create_checkpoint_manager_reconstruction(
                    growth_idx=self.growth_idx,
                    epoch_idx=self.epoch_idx
                )

            # Write final checkpoint.
            checkpoint_saved = self.checkpoint_manager.save(
                checkpoint_number=self.epoch_step_var, check_interval=False
            )

            # Write logs to disk if checkpoint was saved.
            if checkpoint_saved:
                print("Checkpoint saved at {}".format(checkpoint_saved))
                if recon_dict["store_loss_logs"]:
                    self.write_loss_logs()

        if not self.params["export"]["export_every_growth_phase"]:
            # Export final SavedModel for serving.
            self.export_saved_model_reconstruction()

    def training_loop_end_save_model_post_reconstruction(self, training_phase):
        """Saving model when post-reconstruction training loop ends.

        Args:
            training_phase: str, which post-reconstruction training phase
                we're currently training: error_distribution or
                dynamic_threshold.
        """
        # Write final checkpoint.
        checkpoint_saved = self.checkpoint_manager.save(
            checkpoint_number=self.epoch_step_var, check_interval=False
        )

        # Write logs to disk if checkpoint was saved.
        if checkpoint_saved:
            print("Checkpoint saved at {}".format(checkpoint_saved))

        # Export final SavedModel for serving.
        self.export_saved_model_post_reconstruction(training_phase)
