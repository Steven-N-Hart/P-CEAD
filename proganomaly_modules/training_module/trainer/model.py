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

import os
import tensorflow as tf

from . import anomaly
from . import checkpoints
from . import datasets
from . import export
from . import instantiate_model
from . import logs
from . import train
from . import losses
from . import train_step
from . import training_loop


class TrainAndEvaluateModel(
    losses.Losses,
    train.Train,
    checkpoints.Checkpoints,
    logs.Logs,
    instantiate_model.InstantiateModel,
    train_step.TrainStep,
    training_loop.TrainingLoop,
    anomaly.Anomaly,
    export.Export,
    datasets.Datasets
):
    """Train and evaluate loop trainer for model.

    Attributes:
        params: dict, user passed parameters.
        network_objects: dict, instances of `Generator` and `Discriminator`
            network objects.
        optimizers: dict, instances of Keras `Optimizer`s for each network.
        training_phase: tf.Variable, integer to keep track of which training
            phase we're in: {
                0: "reconstruction",
                1: "error_distribution",
                2: "dynamic_threshold"
            }.
        training_phase_schedule: list, schedule of training phases in reverse
            order to be popped off for each requested phase.
        strategy: instance of tf.distribute.strategy.
        global_batch_size_schedule_reconstruction: list, schedule of ints for
            the global batch size after summing batch sizes across replicas.
        global_batch_size_error_distribution: int, error distribution global
            batch size after summing batch sizes across replicas.
        global_batch_size_dynamic_threshold: int, dynamic threshold global
            batch size after summing batch sizes across replicas.
        train_datasets_reconstruction: list, instances of `Dataset` for each
            resolution block for training reconstruction model.
        eval_datasets_reconstruction: list, instances of `Dataset` for each
            resolution block for evaluating reconstruction model.
        train_dataset_error_distribution: `Dataset`, dataset for training
            error distribution model.
        train_dataset_dynamic_threshold: `Dataset`, dataset for training
            dynamic threshold model.
        generator_encoder_phase_needs_real_images: bool, whether the
            generator/encoder training phase needs real images.
        discriminator_train_step_fn: unbound function, function for a
            dicriminator train step using correct strategy and mode.
        generator_train_step_fn: unbound function, function for a
            generator train step using correct strategy and mode.
        error_distribution_train_step_fn: unbound function, function for a
            error distribution train step using correct strategy and mode.
        dynamic_threshold_train_step_fn: unbound function, function for a
            dynamic threshold train step using correct strategy and mode.
        global_step_var: tf.Variable, the global step counter across growths,
            epochs, and steps within epoch.
        growth_step_var: tf.Variable, the growth step counter across epochs and
            steps within epoch.
        epoch_step_var: tf.Variable, the current step through current epoch.
        alpha_var: tf.Variable, used in growth transition network's weighted
            sum, linearly scales through range [0., 1.].
        serving_inputs_Z: tensor, latent Z input to generator for serving
            graph.
        serving_inputs_query_images: list, query image input tensors to
            encoder for serving graph.
        serving_inputs_query_images_names_set: set, strings of query image
            input tensor names for encoder for serving graph.
        serving_outputs: list, output tensors for serving graph.
        serving_model: instance of Keras `Model`, dynamically built model for
            serving based on user provided flags on what to include in the
            export.
        most_recent_export_growth_idx: tf.Variable, most recent export's
            growth index so that there are no repeat exports.
        most_recent_export_epoch_idx: tf.Variable, most recent export's epoch
            index so that there are no repeat exports.
        checkpoint: instance of tf.train.Checkpoint, for saving and restoring
            checkpoints.
        checkpoint_manager: instance of tf.train.CheckpointManager, for
            managing checkpoint path, how often to write, etc.
        summary_file_writer: instance of tf.summary.create_file_writer for
            summaries for TensorBoard.
        num_growths: int, number of growth phases to train over.
        num_steps_until_growth_schedule: list, ints representing a schedule of
            the number of steps/batches until the next growth.
        unique_trainable_variables: dict, list of unique trainable variables
         for each model type unioned across all growths.
        block_idx: int, current growth block/resolution model is in.
        growth_idx: int, current growth index model has progressed to.
        epoch_idx: int, current epoch of growth model is in.
        growth_idx_var: tf.Variable, current growth index model has progressed to.
        epoch_idx_var: tf.Variable, current epoch of growth model is in.
        previous_timestamp: float, the previous timestamp for profiling the
            steps/sec rate.
        losses: dict, the loss of the model at the current step for each
            network.
        loss_logs: dict or list, stores loss over entire training. Dictionary
            if logs are normalized, otherwise list.
        training_loop_restarts: int, current number of times training loop had
            to restart.
        restart_training: bool, whether training needs to restart or not for
            instance due to NaN losses.
        error_distribution_sigma_linv: tf.Variable, the lower triangular
            Choleksy inverse of the error distribution covariance matrix of
            shape (image_depth, image_depth).
        dynamic_threshold: tf.Variable, the dynamic threshold to classify
            pixels as anomalous if above.
    """
    def __init__(self, params):
        """Instantiate trainer.

        Args:
            params: dict, user passed parameters.
        """
        super().__init__()
        self.params = params
        self.assert_params()
        print("model.py: params = {}".format(params))
        generator_dict = self.params["generator"]
        encoder_dict = self.params["encoder"]
        recon_dict = self.params["training"]["reconstruction"]

        self.network_objects = {}
        self.optimizers = {}

        self.training_phase = tf.Variable(
            initial_value=-tf.ones(shape=(), dtype=tf.int64),
            trainable=False,
            name="training_phase"
        )
        self.training_phase_schedule = []

        self.strategy = None
        self.global_batch_size_schedule_reconstruction = []
        self.global_batch_size_error_distribution = 0
        self.global_batch_size_dynamic_threshold = 0

        self.train_datasets_reconstruction = []
        self.eval_datasets_reconstruction = []
        self.train_dataset_error_distribution = None
        self.train_dataset_dynamic_threshold = None

        self.generator_encoder_phase_needs_real_images = any(
            [
                generator_dict["architecture"] == "GANomaly",
                generator_dict["losses"]["berg"][
                    "D_of_G_of_E_of_x_loss_weight"
                ],
                generator_dict["losses"]["berg"][
                    "E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"
                ],
                generator_dict["losses"]["berg"][
                    "E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"
                ],
                generator_dict["losses"]["berg"][
                    "x_minus_G_of_E_of_x_l1_loss_weight"
                ],
                generator_dict["losses"]["berg"][
                    "x_minus_G_of_E_of_x_l2_loss_weight"
                ],
                (
                    encoder_dict["create"] and any(
                        [
                            encoder_dict["losses"]["berg"][
                                "D_of_G_of_E_of_x_loss_weight"
                            ],
                            encoder_dict["losses"]["berg"][
                                "E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"
                            ],
                            encoder_dict["losses"]["berg"][
                                "E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"
                            ],
                            encoder_dict["losses"]["berg"][
                                "x_minus_G_of_E_of_x_l1_loss_weight"
                            ],
                            encoder_dict["losses"]["berg"][
                                "x_minus_G_of_E_of_x_l2_loss_weight"
                            ]
                        ]
                    )
                )
            ]
        )

        self.discriminator_train_step_fn = None
        self.generator_train_step_fn = None
        self.error_distribution_train_step_fn = None
        self.dynamic_threshold_train_step_fn = None

        self.still_training_var = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.int64),
            trainable=False,
            name="still_training_var"
        )

        self.global_step_var = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.int64),
            trainable=False,
            name="global_step"
        )

        self.growth_step_var = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.int64),
            trainable=False,
            name="growth_step"
        )

        self.epoch_step_var = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.int64),
            trainable=False,
            name="epoch_step"
        )

        self.alpha_var = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.float32),
            trainable=False,
            name="alpha_var"
        )

        self.serving_inputs_Z = None
        self.serving_inputs_query_images = []
        self.serving_inputs_query_images_names_set = set()
        self.serving_outputs = []
        self.serving_model = None
        self.most_recent_export_growth_idx = tf.Variable(
            initial_value=self.params["export"]["most_recent_export_growth_idx"],
            trainable=False,
            name="most_recent_export_growth_idx"
        )
        self.most_recent_export_epoch_idx = tf.Variable(
            initial_value=self.params["export"]["most_recent_export_epoch_idx"],
            trainable=False,
            name="most_recent_export_epoch_idx"
        )

        self.checkpoint = None
        self.checkpoint_manager = None

        self.summary_file_writer = None

        # Calculate number of growths. Each progression involves 2 growths,
        # a transition phase and stablization phase.
        self.num_growths = len(self.params["conv_num_filters"]) * 2 - 1
        self.num_steps_until_growth_schedule = (
            recon_dict["num_steps_until_growth_schedule"]
        )

        self.unique_trainable_variables = {}

        self.block_idx = 0

        self.growth_idx_start = (
            recon_dict["initial_growth_idx"]
            if recon_dict["initial_growth_idx"]
            else 0
        )
        self.growth_idx = -1

        self.epoch_idx_start = (
            recon_dict["initial_epoch_idx"]
            if recon_dict["initial_epoch_idx"]
            else 0
        )
        self.epoch_idx = -1

        self.growth_idx_var = tf.Variable(
            initial_value=tf.convert_to_tensor(
                value=recon_dict["initial_growth_idx"],
                dtype=tf.int64
            ),
            trainable=False,
            name="growth_idx"
        )

        self.epoch_idx_var = tf.Variable(
            initial_value=tf.convert_to_tensor(
                value=recon_dict["initial_epoch_idx"],
                dtype=tf.int64
            ),
            trainable=False,
            name="epoch_idx"
        )

        self.previous_timestamp = 0.0

        self.losses = {}
        self.loss_logs = []

        self.training_loop_restarts = 0
        self.restart_training = False

        self.error_distribution_sigma_linv = tf.Variable(
            initial_value=tf.zeros(
                shape=(
                    self.params["training"]["error_distribution"]["image_depth"],
                    self.params["training"]["error_distribution"]["image_depth"]
                ),
                dtype=tf.float32
            ),
            trainable=False,
            name="error_distribution_sigma_linv"
        )

        self.dynamic_threshold = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.float32),
            trainable=False,
            name="dynamic_threshold"
        )

    def assert_params(self):
        """Asserts user passed parameters before we do anything else.
        """
        enc_dict = self.params["encoder"]
        # Ensure architecture layer lists are congruent.
        assert len(self.params["conv_num_filters"]) > 0, (
            "params['conv_num_filters'] must have length greater than zero!"
        )
        assert len(self.params["conv_num_filters"]) == len(self.params["conv_kernel_sizes"]), (
            "len(conv_kernel_sizes) must equal len(conv_num_filters)"
        )
        assert len(self.params["conv_num_filters"]) == len(self.params["conv_strides"]), (
            "len(conv_strides) must equal len(conv_num_filters)"
        )

        # Truncate lists if over the 1024x1024 current limit.
        if len(self.params["conv_num_filters"]) > 9:
            self.params["conv_num_filters"] = self.params["conv_num_filters"][0:10]
            self.params["conv_kernel_sizes"] = self.params["conv_kernel_sizes"][0:10]
            self.params["conv_strides"] = self.params["conv_strides"][0:10]

        # Assert at least one network will be trained, otherwise why have job?
        assert(
            any(
                [
                    self.params["generator"]["train"],
                    enc_dict["create"] and enc_dict["train"],
                    self.params["discriminator"]["create"] and self.params["discriminator"]["train"]
                ]
            )
        ), "At least one network needs to have training enabled!"

        # Assert encoder uses at least one loss, if encoder exists and trains.
        assert (
            any(
                [
                    not enc_dict["create"],
                    not enc_dict["train"],
                    (
                        self.params["generator"]["architecture"] == "berg" and any(
                            [
                                enc_dict["losses"]["berg"]["D_of_G_of_E_of_x_loss_weight"],
                                enc_dict["losses"]["berg"]["D_of_G_of_E_of_G_of_z_loss_weight"],
                                enc_dict["losses"]["berg"]["z_minus_E_of_G_of_z_l1_loss_weight"],
                                enc_dict["losses"]["berg"]["z_minus_E_of_G_of_z_l2_loss_weight"],
                                enc_dict["losses"]["berg"]["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"],
                                enc_dict["losses"]["berg"]["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"],
                                enc_dict["losses"]["berg"]["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"],
                                enc_dict["losses"]["berg"]["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"],
                                enc_dict["losses"]["berg"]["x_minus_G_of_E_of_x_l1_loss_weight"],
                                enc_dict["losses"]["berg"]["x_minus_G_of_E_of_x_l2_loss_weight"]
                            ]
                        )
                    ),
                    (
                        self.params["generator"]["architecture"] == "GANomaly" and any(
                            [
                                enc_dict["losses"]["GANomaly"]["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"],
                                enc_dict["losses"]["GANomaly"]["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"]
                            ]
                        )
                    )
                ]
            )
        ), "Encoder must use at least one loss, if encoder exists and trains!"

        # Can only add uniform noise to z if using subclassed GANomaly models.
        if self.params["generator"]["architecture"] == "GANomaly":
            assert(
                (self.params["generator"]["GANomaly"]["add_uniform_noise_to_z"] and
                self.params["training"]["subclass_models"]) or
                not self.params["generator"]["GANomaly"]["add_uniform_noise_to_z"]
            ), "Can only add uniform noise to z if subclassed models!"

    def prepare_training_components(self):
        """Prepares all components necessary for training.
        """
        # Instantiate model objects.
        self.instantiate_model_objects()

        # Create checkpoint machinery to save/restore checkpoints.
        self.create_checkpoint_machinery()

        # Create summary file writer.
        self.summary_file_writer = tf.summary.create_file_writer(
            logdir=os.path.join(self.params["training"]["output_dir"], "summaries"),
            name="summary_file_writer"
        )

    def train_block_reconstruction(self):
        """Setups training and loops through datasets for reconstruction.
        """
        # Create iterators of datasets.
        self.train_datasets_reconstruction = [
            iter(x) for x in self.train_datasets_reconstruction
        ]
        self.eval_datasets_reconstruction = [
            iter(x) for x in self.eval_datasets_reconstruction
        ]

        # Run training loop.
        max_training_loop_restarts = (
            self.params["training"]["reconstruction"]["max_training_loop_restarts"]
        )
        while self.training_loop_restarts <= max_training_loop_restarts:
            print("Starting reconstruction training loop!")
            self.training_loop_reconstruction()

            # If training loop didn't fail to get back to this scope, break.
            if not self.restart_training:
                break

            # Otherwise, we prematurely returned from the training loop.
            self.training_loop_restarts += 1
            self.restart_training = False
            print(
                "Restarting reconstruction training loop for the {}th time.".format(
                    self.training_loop_restarts
                )
            )

        if self.training_loop_restarts <= max_training_loop_restarts:
            print("Reconstruction training loop complete!")
            if self.training_phase_schedule:
                # Save model at end of training loop.
                self.training_loop_end_save_model_reconstruction()
        else:
            print("Reconstruction training loop exceeded max number of restarts!")

    def train_block_error_distribution(self):
        """Setups training and loops through datasets for error distribution.
        """
        # Create iterators of datasets.
        self.train_dataset_error_distribution = (
            iter(self.train_dataset_error_distribution)
        )

        # Run training loop.
        max_training_loop_restarts = (
            self.params["training"]["error_distribution"]["max_training_loop_restarts"]
        )
        while self.training_loop_restarts <= max_training_loop_restarts:
            print("Starting error distribution training loop!")
            self.training_loop_error_distribution()

            # If training loop didn't fail to get back to this scope, break.
            if not self.restart_training:
                break

            # Otherwise, we prematurely returned from the training loop.
            self.training_loop_restarts += 1
            self.restart_training = False
            print(
                "Restarting error distribution training loop for the {}th time.".format(
                    self.training_loop_restarts
                )
            )

        if self.training_loop_restarts <= max_training_loop_restarts:
            print("Error distribution training loop complete!")
            if self.training_phase_schedule:
                # Save model at end of training loop.
                self.training_loop_end_save_model_post_reconstruction(
                    training_phase="error_distribution"
                )
        else:
            print("Error distribution training loop exceeded max number of restarts!")

    def train_block_dynamic_threshold(self):
        """Setups training and loops through datasets for dynamic thresholds.
        """
        # Create iterators of datasets.
        self.train_dataset_dynamic_threshold = (
            iter(self.train_dataset_dynamic_threshold)
        )

        # Run training loop.
        max_training_loop_restarts = (
            self.params["training"]["dynamic_threshold"]["max_training_loop_restarts"]
        )
        while self.training_loop_restarts <= max_training_loop_restarts:
            print("Starting dynamic threshold training loop!")
            self.training_loop_dynamic_threshold()

            # If training loop didn't fail to get back to this scope, break.
            if not self.restart_training:
                break

            # Otherwise, we prematurely returned from the training loop.
            self.training_loop_restarts += 1
            self.restart_training = False
            print(
                "Restarting dynamic threshold training loop for the {}th time.".format(
                    self.training_loop_restarts
                )
            )

        if self.training_loop_restarts <= max_training_loop_restarts:
            print("Dynamic threshold training loop complete!")
        else:
            print("Dynamic threshold training loop exceeded max number of restarts!")

    @tf.function
    def training_started(self):
        """Assigns still training variable to indicate training has begun.
        """
        self.still_training_var.assign(
            value=tf.ones(shape=(), dtype=tf.int64)
        )

    @tf.function
    def training_complete(self):
        """Assigns still training variable to indicate training has completed.
        """
        self.still_training_var.assign(
            value=tf.zeros(shape=(), dtype=tf.int64)
        )

    def train_blocks(self):
        """Setups training and loops through datasets for each training phase.
        """
        # Instantiate models, create checkpoints, create summary file writer.
        self.prepare_training_components()

        # Set variable to mark that training has started.
        self.training_started()

        print("training_phase_schedule = {}".format(self.training_phase_schedule))
        while self.training_phase_schedule:
            training_phase = self.training_phase_schedule.pop()

            if training_phase == -1:
                self.train_block_reconstruction()

            if training_phase == 0:
                self.train_block_error_distribution()

            if training_phase == 1:
                self.train_block_dynamic_threshold()

        # Training is complete so we can safely clear variable.
        self.training_complete()

        print("All training complete! Saving final checkpoint and SavedModel!")
        if self.training_phase.numpy() == 0:
            self.training_loop_end_save_model_reconstruction()
        elif self.training_phase.numpy() == 1:
            self.training_loop_end_save_model_post_reconstruction(
                training_phase="error_distribution"
            )
        elif self.training_phase.numpy() == 2:
            self.training_loop_end_save_model_post_reconstruction(
                training_phase="dynamic_threshold"
            )

    def train_and_evaluate(self):
        """Trains and evaluates Keras model.

        Args:
            args: dict, user passed parameters.

        Returns:
            Generator's `Model` object for in-memory predictions.
        """
        reconstruct_dict = self.params["training"]["reconstruction"]
        error_dict = self.params["training"]["error_distribution"]
        threshold_dict = self.params["training"]["dynamic_threshold"]
        if self.params["training"]["distribution_strategy"]:
            # If the list of devices is not specified in the
            # Strategy constructor, it will be auto-detected.
            if self.params["training"]["distribution_strategy"] == "Mirrored":
                self.strategy = tf.distribute.MirroredStrategy()
            num_replicas = self.strategy.num_replicas_in_sync
            print("Number of devices = {}".format(num_replicas))

            # Set global batch size for training phases.
            self.global_batch_size_schedule_reconstruction = [
                x * num_replicas
                for x in reconstruct_dict["train_batch_size_schedule"]
            ]

            self.global_batch_size_error_distribution = (
                error_dict["train_batch_size"] * num_replicas
            )

            self.global_batch_size_dynamic_threshold = (
                threshold_dict["train_batch_size"] * num_replicas
            )

            # Shorten growth schedule due to parallel work from replicas.
            self.num_steps_until_growth_schedule = [
                x // num_replicas
                for x in reconstruct_dict["num_steps_until_growth_schedule"]
            ]

            # Get input datasets. Batch size is split evenly between replicas.
            self.get_all_datasets()

            with self.strategy.scope():
                # Training block setups training, then loops through datasets.
                self.train_blocks()

        else:
            # Set global batch size for training phases.
            self.global_batch_size_schedule_reconstruction = (
                reconstruct_dict["train_batch_size_schedule"]
            )

            self.global_batch_size_error_distribution = (
                error_dict["train_batch_size"]
            )

            self.global_batch_size_dynamic_threshold = (
                threshold_dict["train_batch_size"]
            )

            # Get input datasets.
            self.get_all_datasets()

            # Training block setups training, then loops through datasets.
            self.train_blocks()
