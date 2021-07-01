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

from . import batch_calculate_distribution_statistics
from . import discriminators
from . import generators


class InstantiateModel(object):
    """Class used for instantiating model objects.
    """
    def __init__(self):
        """Instantiate instance of `InstantiateModel`.
        """
        pass

    def _instantiate_optimizer(self, scope):
        """Instantiates scoped optimizer with parameters.

        Args:
            scope: str, the name of the network of interest.
        """
        # Create optimizer map.
        optimizers = {
            "Adadelta": tf.keras.optimizers.Adadelta,
            "Adagrad": tf.keras.optimizers.Adagrad,
            "Adam": tf.keras.optimizers.Adam,
            "Adamax": tf.keras.optimizers.Adamax,
            "Ftrl": tf.keras.optimizers.Ftrl,
            "Nadam": tf.keras.optimizers.Nadam,
            "RMSprop": tf.keras.optimizers.RMSprop,
            "SGD": tf.keras.optimizers.SGD
        }

        # Get optimizer and instantiate it.
        if self.params[scope]["optimizer"] == "Adam":
            optimizer = optimizers[self.params[scope]["optimizer"]](
                learning_rate=self.params[scope]["learning_rate"],
                beta_1=self.params[scope]["adam_beta1"],
                beta_2=self.params[scope]["adam_beta2"],
                epsilon=self.params[scope]["adam_epsilon"],
                name="{}_{}_optimizer".format(
                    scope, self.params[scope]["optimizer"].lower()
                )
            )
        else:
            optimizer = optimizers[self.params[scope]["optimizer"]](
                learning_rate=self.params[scope]["learning_rate"],
                name="{}_{}_optimizer".format(
                    scope, self.params[scope]["optimizer"].lower()
                )
            )

        self.optimizers[scope] = optimizer

    def _instantiate_optimizers(self):
        """Instantiates all network optimizers.
        """
        # Instantiate optimizers.
        self._instantiate_optimizer(scope="generator")
        self._instantiate_optimizer(scope="encoder")
        self._instantiate_optimizer(scope="discriminator")

    def _instantiate_network_objects(self):
        """Instantiates generator and discriminator objects with parameters.
        """
        # Instantiate generator.
        if self.params["generator"]["architecture"] == "berg":
            generator_type = "decoder"
        elif self.params["generator"]["architecture"] == "GANomaly":
            generator_type = "unet"

        if self.params["training"]["subclass_models"]:
            generator_class = generators.GeneratorsSubClass
        else:
            generator_class = generators.GeneratorsFunctional

        self.network_objects["generator"] = generator_class(
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["generator"]["l1_regularization_scale"],
                l2=self.params["generator"]["l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="generator",
            params=self.params,
            alpha_var=self.alpha_var,
            num_growths=self.num_growths,
            network_type=generator_type
        )

        if self.params["encoder"]["create"]:
            # Instantiate encoder.
            if self.params["training"]["subclass_models"]:
                encoder_class = discriminators.DiscriminatorsSubClass
            else:
                encoder_class = discriminators.DiscriminatorsFunctional

            self.network_objects["encoder"] = encoder_class(
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=self.params["encoder"]["l1_regularization_scale"],
                    l2=self.params["encoder"]["l2_regularization_scale"]
                ),
                bias_regularizer=None,
                name="encoder",
                params=self.params,
                alpha_var=self.alpha_var,
                num_growths=self.num_growths,
                network_type="encoder"
            )

        if self.params["discriminator"]["create"]:
            # Instantiate discriminator.
            if self.params["training"]["subclass_models"]:
                discriminator_class = discriminators.DiscriminatorsSubClass
            else:
                discriminator_class = discriminators.DiscriminatorsFunctional

            self.network_objects["discriminator"] = discriminator_class(
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=self.params["discriminator"][
                        "l1_regularization_scale"
                    ],
                    l2=self.params["discriminator"][
                        "l2_regularization_scale"
                    ]
                ),
                bias_regularizer=None,
                name="discriminator",
                params=self.params,
                alpha_var=self.alpha_var,
                num_growths=self.num_growths,
                network_type="discriminator"
            )

        train_dict = self.params["training"]
        if train_dict["train_error_distribution"]:
            # Instantiate error distribution object.
            self.network_objects["error_distribution"] = (
                batch_calculate_distribution_statistics.BatchCalculateDistributionStatistics(
                    params=train_dict["error_distribution"],
                    num_cols=train_dict["error_distribution"]["image_depth"]
                )
            )

        if train_dict["train_dynamic_threshold"]:
            if train_dict["dynamic_threshold"]["use_supervised"]:
                # TODO: Add this later.
                raise NotImplementedError
            else:
                # Instantiate error distribution object.
                self.network_objects["dynamic_threshold"] = (
                    batch_calculate_distribution_statistics.BatchCalculateDistributionStatistics(
                        params=train_dict["dynamic_threshold"]["unsupervised"],
                        num_cols=1
                    )
                )

    def _get_unique_trainable_variables(self, scope):
        """Gets union of unique trainable variables within given scope.

        Args:
            scope: str, the name of the network of interest.
        """
        # All names of 0th model variables are already guaranteed unique.
        unique_names = set(
            [
                var.name
                for var in (
                    self.network_objects[scope].models[0].trainable_variables
                )
            ]
        )

        unique_trainable_variables = (
            self.network_objects[scope].models[0].trainable_variables
        )

        # Loop through future growth models to get trainable variables.
        for i in range(1, self.num_growths):
            trainable_variables = (
                self.network_objects[scope].models[i].trainable_variables
            )

            # Loop through variables and append any that are unique.
            for var in trainable_variables:
                if var.name not in unique_names:
                    unique_names.add(var.name)
                    unique_trainable_variables.append(var)

        self.unique_trainable_variables[scope] = unique_trainable_variables

    def _create_optimizer_variable_slots(self, scope):
        """Creates optimizer variable slots for given scoped model type.

        It is needed to build any optimizer variables within graph mode since
        variables cannot be created outside the first call of a tf.function.

        Args:
            scope: str, the name of the network of interest.
        """
        # Get the union of all trainable variables across all model growths.
        self._get_unique_trainable_variables(scope)

        # Create placeholder gradients that we can apply to model variables.
        # Note: normally some gradients (especially of future growth models)
        placeholder_gradients = [
            tf.zeros_like(input=var, dtype=tf.float32)
            for var in self.unique_trainable_variables[scope]
        ]

        # Apply gradients to create optimizer variable slots for each
        # trainable variable.
        self.optimizers[scope].apply_gradients(
            zip(
                placeholder_gradients, self.unique_trainable_variables[scope]
            )
        )

    @tf.function
    def _non_distributed_instantiate_optimizer_variables(self):
        """Instantiates optimizer variable slots for given scoped model type.

        It is needed to build any optimizer variables within graph mode since
        variables cannot be created outside the first call of a tf.function.
        This is the non-distributed version.

        Args:
            scope: str, the name of the network of interest.
        """
        self._create_optimizer_variable_slots(scope="generator")
        if self.params["encoder"]["create"]:
            self._create_optimizer_variable_slots(scope="encoder")
        if self.params["discriminator"]["create"]:
            self._create_optimizer_variable_slots(scope="discriminator")

        return tf.zeros(shape=(), dtype=tf.float32)

    @tf.function
    def _distributed_instantiate_optimizer_variables(self):
        """Instantiates optimizer variable slots for given scoped model type.

        It is needed to build any optimizer variables within graph mode since
        variables cannot be created outside the first call of a tf.function.
        This is the distributed version.

        Args:
            scope: str, the name of the network of interest.
        """
        if self.params["training"]["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self._non_distributed_instantiate_optimizer_variables
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def instantiate_model_objects(self):
        """Instantiate model network objects, network models, and optimizers.
        """
        # Instantiate generator and discriminator optimizers.
        self._instantiate_optimizers()

        # Instantiate generator and discriminator objects.
        self._instantiate_network_objects()

        # Instantiate optimizer variable slots.
        if self.strategy:
            _ = self._distributed_instantiate_optimizer_variables()
        else:
            _ = self._non_distributed_instantiate_optimizer_variables()
