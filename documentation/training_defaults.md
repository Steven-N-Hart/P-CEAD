# Proganomaly Training Pipeline

## License

Copyright 2020 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at [Apache License Page](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

## Training Parameters

- Dictionary of generator configs.
######
        generator_dict = dict()

- Which paper to use for generator architecture: "berg", "GANomaly".
######
        generator_dict["architecture"] = "GANomaly"

- Whether generator will be trained or not.
######
        generator_dict["train"] = True

- Number of steps to train generator for per cycle.
######
        generator_dict["train_steps"] = 1

- Whether to normalize latent vector before projection.
######
        generator_dict["normalize_latents"] = True
    
- Whether to use pixel norm op after each convolution.
######
        generator_dict["use_pixel_norm"] = True
    
- Small value to add to denominator for numerical stability.
######
        generator_dict["pixel_norm_epsilon"] = 1e-8
    
- The 3D dimensions to project latent noise vector into.
######
        generator_dict["projection_dims"] = [4, 4, 512]
    
- The amount of leakyness of generator's leaky relus.
######
        generator_dict["leaky_relu_alpha"] = 0.2
    
- The final activation function of generator: None, sigmoid, tanh, relu.
######
        generator_dict["final_activation"] = "None"
    
- Scale factor for L1 regularization for generator.
######
        generator_dict["l1_regularization_scale"] = 0.
    
- Scale factor for L2 regularization for generator.
######
        generator_dict["l2_regularization_scale"] = 0.
    
- Name of optimizer to use for generator.
######
        generator_dict["optimizer"] = "Adam"

- How quickly we train model by scaling the gradient for generator.
######
        generator_dict["learning_rate"] = 0.001

- Adam optimizer's beta1 hyperparameter for first moment.
######
        generator_dict["adam_beta1"] = 0.0
    
- Adam optimizer's beta2 hyperparameter for second moment.
######
        generator_dict["adam_beta2"] = 0.99
    
- Adam optimizer's epsilon hyperparameter for numerical stability.
######
        generator_dict["adam_epsilon"] = 1e-8
    
- Global clipping to prevent gradient norm to exceed this value for generator.
######
        generator_dict["clip_gradients"] = None

- Dictionary of generator configs in Berg's architecture.
######
        generator_berg_dict = dict()
    
- Dictionary of generator configs in GANomaly's architecture.
######
        generator_ganomaly_dict = dict()

- Dictionary of generator losses configs in Berg's architecture.
######
        generator_berg_losses_dict = dict()
    
- Dictionary of generator losses configs in GANomaly's architecture.
######
        generator_ganomaly_losses_dict = dict()

- The latent size of the noise vector.
######
        generator_berg_dict["latent_size"] = 512

- The latent vector's random normal mean.
######
        generator_berg_dict["latent_mean"] = 0.0
        
- The latent vector's random normal standard deviation.
######
        generator_berg_dict["latent_stddev"] = 1.0

- These are just example values, yours will vary.
    - Weights to multiply loss of D(G(z))
######
        generator_berg_losses_dict["D_of_G_of_z_loss_weight"] = 1.0

- Weights to multiply loss of D(G(E(x)))
######
        generator_berg_losses_dict["D_of_G_of_E_of_x_loss_weight"] = 0.0
        
- Weights to multiply loss of D(G(E(G(z)))
######
        generator_berg_losses_dict["D_of_G_of_E_of_G_of_z_loss_weight"] = 0.0

- Weights to multiply loss of z - E(G(z))
######
        generator_berg_losses_dict["z_minus_E_of_G_of_z_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["z_minus_E_of_G_of_z_l2_loss_weight"] = 0.0
        
- Weights to multiply loss of G(z) - G(E(G(z))
######
        generator_berg_losses_dict["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"] = 0.0
        
- Weights to multiply loss of E(x) - E(G(E(x)))
######
        generator_berg_losses_dict["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"] = 1.0
        generator_berg_losses_dict["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"] = 0.0
        
- Weights to multiply loss of x - G(E(x))
######
        generator_berg_losses_dict["x_minus_G_of_E_of_x_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["x_minus_G_of_E_of_x_l2_loss_weight"] = 0.0

- GANomaly parameters to zero.
    - Weights to multiply loss of D(G(x))
######
        generator_ganomaly_losses_dict["D_of_G_of_x_loss_weight"] = 0.0

- Weights to multiply loss of x - G(x)
######
        generator_ganomaly_losses_dict["x_minus_G_of_x_l1_loss_weight"] = 0.0
        generator_ganomaly_losses_dict["x_minus_G_of_x_l2_loss_weight"] = 0.0

- Weights to multiply loss of Ge(x) - E(G(x))
######
        generator_ganomaly_losses_dict["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"] = 0.0
        generator_ganomaly_losses_dict["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"] = 0.0

- Whether generator GANomaly architecture uses U-net skip connection for each block.
######
        generator_ganomaly_dict["use_unet_skip_connections"] = [True] * 9

- Percent of masking image inputs to generator.
######
        generator_ganomaly_dict["mask_generator_input_images_percent"] = 0.2
        
- Integer amount to randomly shift image mask block sizes.
######
        generator_ganomaly_dict["image_mask_block_random_shift_amount"] = 0
        
- Whether to use shuffle or dead image block masking.
######
        generator_ganomaly_dict["use_shuffle_image_masks"] = True
        
- Whether to add uniform noise to GANomaly Z vector.
######
        generator_ganomaly_dict["add_uniform_noise_to_z"] = True
        
- Whether to add uniform noise to fake images.
######
        generator_ganomaly_dict["add_uniform_noise_to_fake_images"] = True

- These are just example values, yours will vary.
    - Weights to multiply loss of D(G(x))
######
        generator_ganomaly_losses_dict["D_of_G_of_x_loss_weight"] = 1.0

- Weights to multiply loss of x - G(x)
######
        generator_ganomaly_losses_dict["x_minus_G_of_x_l1_loss_weight"] = 0.0
        generator_ganomaly_losses_dict["x_minus_G_of_x_l2_loss_weight"] = 100.0

- Weights to multiply loss of Ge(x) - E(G(x))
######
        generator_ganomaly_losses_dict["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"] = 0.0
        generator_ganomaly_losses_dict["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"] = 0.0

- Berg parameters to zero.
    - Weights to multiply loss of D(G(z))
######
        generator_berg_losses_dict["D_of_G_of_z_loss_weight"] = 0.0

- Weights to multiply loss of D(G(E(x)))
######
        generator_berg_losses_dict["D_of_G_of_E_of_x_loss_weight"] = 0.0

- Weights to multiply loss of D(G(E(G(z)))
######
        generator_berg_losses_dict["D_of_G_of_E_of_G_of_z_loss_weight"] = 0.0

- Weights to multiply loss of z - E(G(z))
######
        generator_berg_losses_dict["z_minus_E_of_G_of_z_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["z_minus_E_of_G_of_z_l2_loss_weight"] = 0.0

- Weights to multiply loss of G(z) - G(E(G(z))
######
        generator_berg_losses_dict["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"] = 0.0
        
- Weights to multiply loss of E(x) - E(G(E(x)))
######
        generator_berg_losses_dict["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"] = 0.0
        
- Weights to multiply loss of x - G(E(x))
######
        generator_berg_losses_dict["x_minus_G_of_E_of_x_l1_loss_weight"] = 0.0
        generator_berg_losses_dict["x_minus_G_of_E_of_x_l2_loss_weight"] = 0.0


- Dictionary of encoder configs.
######
        encoder_dict = dict()

- These are optional if using GANomaly architecture, required for berg.
    - Whether encoder will be created or not.
######
        encoder_dict["create"] = True
- Whether encoder will be trained or not.
######
        encoder_dict["train"] = True

- Whether to use minibatch stddev op before first base conv layer.
######
            encoder_dict["use_minibatch_stddev"] = True
    
- The size of groups to split minibatch examples into.
######
        encoder_dict["minibatch_stddev_group_size"] = 4
    
- Whether to average across feature maps and pixels for minibatch stddev.
######
        encoder_dict["minibatch_stddev_use_averaging"] = True
    
- The amount of leakyness of encoder's leaky relus.
######
        encoder_dict["leaky_relu_alpha"] = 0.2
    
- Scale factor for L1 regularization for encoder.
######
        encoder_dict["l1_regularization_scale"] = 0.
    
- Scale factor for L2 regularization for encoder.
######
        encoder_dict["l2_regularization_scale"] = 0.
    
- Name of optimizer to use for encoder.
######
        encoder_dict["optimizer"] = "Adam"
        
- How quickly we train model by scaling the gradient for encoder.
######
        encoder_dict["learning_rate"] = 0.001
        
- Adam optimizer's beta1 hyperparameter for first moment.
######
        encoder_dict["adam_beta1"] = 0.0
    
- Adam optimizer's beta2 hyperparameter for second moment.
######
        encoder_dict["adam_beta2"] = 0.99
    
- Adam optimizer's epsilon hyperparameter for numerical stability.
######
        encoder_dict["adam_epsilon"] = 1e-8
    
- Global clipping to prevent gradient norm to exceed this value for encoder.
######
        encoder_dict["clip_gradients"] = None

- Dictionary of encoder losses config
######
        encoder_losses_dict = dict()

- Dictionary of encoder losses config with Berg's architecture
######
        encoder_losses_berg_dict = dict()
    
- Weights to multiply loss of D(G(E(x)))
######
        encoder_losses_berg_dict["D_of_G_of_E_of_x_loss_weight"] = 0.0
    
- Weights to multiply loss of D(G(E(G(z)))
######
        encoder_losses_berg_dict["D_of_G_of_E_of_G_of_z_loss_weight"] = 0.0

- Weights to multiply loss of z - E(G(z))
######
        encoder_losses_berg_dict["z_minus_E_of_G_of_z_l1_loss_weight"] = 0.0
        encoder_losses_berg_dict["z_minus_E_of_G_of_z_l2_loss_weight"] = 0.0
    
- Weights to multiply loss of G(z) - G(E(G(z))
######
        encoder_losses_berg_dict["G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight"] = 0.0
        encoder_losses_berg_dict["G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight"] = 0.0
    
- Weights to multiply loss of E(x) - E(G(E(x)))
######
        encoder_losses_berg_dict["E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight"] = 0.0
        encoder_losses_berg_dict["E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight"] = 0.0

- Weights to multiply loss of x - G(E(x))
######
        encoder_losses_berg_dict["x_minus_G_of_E_of_x_l1_loss_weight"] = 0.0
        encoder_losses_berg_dict["x_minus_G_of_E_of_x_l2_loss_weight"] = 0.0

- Dictionary of encoder losses config with GANomaly's architecture
######
        encoder_losses_ganomaly_dict = dict()

- Weights to multiply loss of Ge(x) - E(G(x))
######
        encoder_losses_ganomaly_dict["Ge_of_x_minus_E_of_G_of_x_l1_loss_weight"] = 0.0
        encoder_losses_ganomaly_dict["Ge_of_x_minus_E_of_G_of_x_l2_loss_weight"] = 1.0

- Dictionary of discriminator configs.
######
        discriminator_dict = dict()

- Whether discriminator will be created or not.
######
        discriminator_dict["create"] = True
    
- Whether discriminator will be trained or not.
######
        discriminator_dict["train"] = True
    
- Number of steps to train discriminator for per cycle.
######
        discriminator_dict["train_steps"] = 1

- Whether to use minibatch stddev op before first base conv layer.
######
        discriminator_dict["use_minibatch_stddev"] = True
    
- The size of groups to split minibatch examples into.
######
        discriminator_dict["minibatch_stddev_group_size"] = 4
    
- Whether to average across feature maps and pixels for minibatch stddev.
######
        discriminator_dict["minibatch_stddev_use_averaging"] = True
    
- The amount of leakyness of discriminator's leaky relus.
######
        discriminator_dict["leaky_relu_alpha"] = 0.2
    
- Scale factor for L1 regularization for discriminator.
######
        discriminator_dict["l1_regularization_scale"] = 0.
    
- Scale factor for L2 regularization for discriminator.
######
        discriminator_dict["l2_regularization_scale"] = 0.
    
- Name of optimizer to use for discriminator.
######
        discriminator_dict["optimizer"] = "Adam"
    
- How quickly we train model by scaling the gradient for discriminator.
######
        discriminator_dict["learning_rate"] = 0.001
    
- Adam optimizer's beta1 hyperparameter for first moment.
######
        discriminator_dict["adam_beta1"] = 0.0
    
- Adam optimizer's beta2 hyperparameter for second moment.
######
        discriminator_dict["adam_beta2"] = 0.99
    
- Adam optimizer's epsilon hyperparameter for numerical stability.
######
        discriminator_dict["adam_epsilon"] = 1e-8
    
- Global clipping to prevent gradient norm to exceed this value for discriminator.
######
        discriminator_dict["clip_gradients"] = None
    
- Coefficient of gradient penalty for discriminator.
######
        discriminator_dict["gradient_penalty_coefficient"] = 10.0

- Target value of gradient magnitudes for gradient penalty for discriminator.
######
        discriminator_dict["gradient_penalty_target"] = 1.0
    
- Coefficient of epsilon drift penalty for discriminator.
######
        discriminator_dict["epsilon_drift"] = 0.001

- Dictionary of discriminator losses config
######
        discriminator_losses_dict = dict()
    
- Weight to multiply loss of D(x)
######
        discriminator_losses_dict["D_of_x_loss_weight"] = 1.0

- Dictionary of discriminator losses config with Berg's architecture
######
        discriminator_losses_berg_dict = dict()
    
- Weight to multiply loss of D(G(z))
######
        discriminator_losses_berg_dict["D_of_G_of_z_loss_weight"] = 0.0
    
- Weight to multiply loss of D(G(E(x)))
######
        discriminator_losses_berg_dict["D_of_G_of_E_of_x_loss_weight"] = 0.0
    
- Weight to multiply loss of D(G(E(G(z)))
######
        discriminator_losses_berg_dict["D_of_G_of_E_of_G_of_z_loss_weight"] = 0.0

- Dictionary of discriminator losses config with GANomaly's architecture
######
        discriminator_losses_ganomaly_dict = dict()
 
- Weight to multiply loss of D(G(x))
######
        discriminator_losses_ganomaly_dict["D_of_G_of_x_loss_weight"] = 1.0

- Dictionary of reconstruction configs.
######
        reconstruction_dict = dict()

- Whether using multiple resolutions across a list of TF Records.
######
        reconstruction_dict["use_multiple_resolution_records"] = True
    
- GCS locations to read reconstruction training data.
######
        reconstruction_dict["train_file_patterns"] = [
            "data/cifar10_car/train_{0}x{0}_*.tfrecord".format(4 * 2 ** i)
            for i in range(4)
        ]
    
- GCS locations to read reconstruction evaluation data.
######
        reconstruction_dict["eval_file_patterns"] = [
            "data/cifar10_car/test_{0}x{0}_*.tfrecord".format(4 * 2 ** i)
            for i in range(4)
        ]
    
- Which dataset to use for reconstruction training:
    - "mnist", "cifar10", "cifar10_car", "tf_record"
######
        reconstruction_dict["dataset"] = "tf_record"
    
- TF Record Example feature schema for reconstruction.
######
        reconstruction_dict["tf_record_example_schema"] = [
            {
                "name": "image_raw",
                "type": "FixedLen",
                "shape": [],
                "dtype": "str"
            },
            {
                "name": "label",
                "type": "FixedLen",
                "shape": [],
                "dtype": "int"
            }
        ]
    
- Name of image feature within schema dictionary.
######
        reconstruction_dict["image_feature_name"] = "image_raw"
    
- Encoding of image: raw, png, or jpeg.
######
        reconstruction_dict["image_encoding"] = "raw"
    
- Height of predownscaled image if NOT using multiple resolution records.
######
        reconstruction_dict["image_predownscaled_height"] = 32
    
- Width of predownscaled image if NOT using multiple resolution records.
######
        reconstruction_dict["image_predownscaled_width"] = 32
    
- Depth of image, number of channels.
######
        reconstruction_dict["image_depth"] = 3
    
- Name of label feature within schema dictionary.
######
        reconstruction_dict["label_feature_name"] = "label"
    
- Schedule list of number of epochs to train for reconstruction.
######
        reconstruction_dict["num_epochs_schedule"] = [1] * 9
    
- Number of examples in one epoch of reconstruction training set.
######
        reconstruction_dict["train_dataset_length"] = 400
    
- Schedule list of number of examples in reconstruction training batch for each resolution block.
######
        reconstruction_dict["train_batch_size_schedule"] = [4] * 9
    
- Schedule list of number of examples in reconstruction evaluation batch for each resolution block.
######
        reconstruction_dict["eval_batch_size_schedule"] = [4] * 9
    
- Number of steps/batches to evaluate for reconstruction.
######
        reconstruction_dict["eval_steps"] = 1
    
- List of number of examples until block added to networks.
######
        reconstruction_dict["num_examples_until_growth_schedule"] = [
            epochs * reconstruction_dict["train_dataset_length"]
            for epochs in reconstruction_dict["num_epochs_schedule"]
        ]
    
- List of number of steps/batches until block added to networks.
######
        reconstruction_dict["num_steps_until_growth_schedule"] = [
            ex // bs
            for ex, bs in zip(
                reconstruction_dict["num_examples_until_growth_schedule"],
                reconstruction_dict["train_batch_size_schedule"]
            )
        ]
    
- Whether to autotune input function performance for reconstruction datasets.
######
        reconstruction_dict["input_fn_autotune"] = True
    
- How many steps to train before writing steps and loss to log.
######
        reconstruction_dict["log_step_count_steps"] = 10
    
- How many steps to train before saving a summary.
######
        reconstruction_dict["save_summary_steps"] = 10
    
- Whether to write loss summaries for TensorBoard.
######
        reconstruction_dict["write_loss_summaries"] = False
    
- Whether to write generator image summaries for TensorBoard.
######
        reconstruction_dict["write_generator_image_summaries"] = False
    
- Whether to write encoder image summaries for TensorBoard.
######
        reconstruction_dict["write_encoder_image_summaries"] = False
    
- Whether to write variable histogram summaries for TensorBoard.
######
        reconstruction_dict["write_variable_histogram_summaries"] = False
    
- Whether to write gradient histogram summaries for TensorBoard.
######
        reconstruction_dict["write_gradient_histogram_summaries"] = False
    
- How many steps to train reconstruction before saving a checkpoint.
######
        reconstruction_dict["save_checkpoints_steps"] = 10000
    
- Max number of reconstruction checkpoints to keep.
######
        reconstruction_dict["keep_checkpoint_max"] = 10
    
- Whether to save checkpoint every growth phase.
######
        reconstruction_dict["checkpoint_every_growth_phase"] = True
    
- Whether to save checkpoint every epoch.
######
        reconstruction_dict["checkpoint_every_epoch"] = True
    
- Checkpoint growth index to restore checkpoint.
######
        reconstruction_dict["checkpoint_growth_idx"] = 0
    
- Checkpoint epoch index to restore checkpoint.
######
        reconstruction_dict["checkpoint_epoch_idx"] = 0
    
- The checkpoint save path for saving and restoring.
######
        reconstruction_dict["checkpoint_save_path"] = ""
    
- Whether to store loss logs.
######
        reconstruction_dict["store_loss_logs"] = True
    
- Whether to normalize loss logs.
######
        reconstruction_dict["normalized_loss_logs"] = True
    
- Whether to print model summaries.
######
        reconstruction_dict["print_training_model_summaries"] = False
    
- Initial growth index to resume training midway.
######
        reconstruction_dict["initial_growth_idx"] = 0
        
- Initial epoch index to resume training midway.
######
        reconstruction_dict["initial_epoch_idx"] = 0
    
- Max number of times training loop can be restarted such as for NaN losses.
######
        reconstruction_dict["max_training_loop_restarts"] = 10

- Whether to scale layer weights to equalize learning rate each forward pass.
######
        reconstruction_dict["use_equalized_learning_rate"] = True
    
- Whether to normalize reconstruction losses by number of pixels.
######
        reconstruction_dict["normalize_reconstruction_losses"] = True

- Dictionary of error_distribution configs.
######
        error_distribution_dict = dict()

- Whether using multiple resolutions across a list of TF Records.
######
        error_distribution_dict["use_multiple_resolution_records"] = False
    
- GCS locations to read error distribution training data.
######
        error_distribution_dict["train_file_pattern"] = "data/cifar10_car/train_32x32_*.tfrecord"
    
- GCS locations to read error distribution training data.
######
        error_distribution_dict["eval_file_pattern"] = "data/cifar10_car/train_32x32_*.tfrecord"
    
- Which dataset to use for error distribution training:
    - "mnist", "cifar10", "cifar10_car", "tf_record"
######
        error_distribution_dict["dataset"] = "tf_record"
    
- TF Record Example feature schema for error distribution.
######
        error_distribution_dict["tf_record_example_schema"] = [
            {
                "name": "image_raw",
                "type": "FixedLen",
                "shape": [],
                "dtype": "str"
            },
            {
                "name": "label",
                "type": "FixedLen",
                "shape": [],
                "dtype": "int"
            }
        ]
    
- Name of image feature within schema dictionary.
######
        error_distribution_dict["image_feature_name"] = "image_raw"
    
- Encoding of image: raw, png, or jpeg.
######
        error_distribution_dict["image_encoding"] = "raw"
    
- Height of predownscaled image if NOT using multiple resolution records.
######
        error_distribution_dict["image_predownscaled_height"] = 32
    
- Width of predownscaled image if NOT using multiple resolution records.
######
        error_distribution_dict["image_predownscaled_width"] = 32
    
- Depth of image, number of channels.
######
        error_distribution_dict["image_depth"] = 3
    
- Name of label feature within schema dictionary.
######
        error_distribution_dict["label_feature_name"] = "label"
    
- Number of examples in one epoch of error distribution training set.
######
        error_distribution_dict["train_dataset_length"] = 400
    
- Number of examples in error distribution training batch.
######
        error_distribution_dict["train_batch_size"] = 32
    
- Number of steps/batches to evaluate for error distribution.
######
        error_distribution_dict["eval_steps"] = 10
    
- Whether to autotune input function performance for error distribution datasets.
######
        error_distribution_dict["input_fn_autotune"] = True
    
- How many steps to train error distribution before saving a checkpoint.
######
        error_distribution_dict["save_checkpoints_steps"] = 10000
    
- Max number of error distribution checkpoints to keep.
######
        error_distribution_dict["keep_checkpoint_max"] = 10
    
- Max number of times training loop can be restarted.
######
        error_distribution_dict["max_training_loop_restarts"] = 10

- Whether using sample or population covariance for error distribution.
######
        error_distribution_dict["use_sample_covariance"] = True

- Dictionary of dynamic_threshold configs.
######
        dynamic_threshold_dict = dict()

- Whether using multiple resolutions across a list of TF Records.
######
        dynamic_threshold_dict["use_multiple_resolution_records"] = False
    
- GCS locations to read dynamic threshold training data.
######
        dynamic_threshold_dict["train_file_pattern"] = "data/cifar10_car/train_32x32_*.tfrecord"
    
- GCS locations to read dynamic threshold evaluation data.
######
        dynamic_threshold_dict["eval_file_pattern"] = "data/cifar10_car/train_32x32_*.tfrecord"
    
- Which dataset to use for dynamic threshold training:
    - "mnist", "cifar10", "cifar10_car", "tf_record"
######
        dynamic_threshold_dict["dataset"] = "tf_record"
    
- TF Record Example feature schema for dynamic threshold.
######
        dynamic_threshold_dict["tf_record_example_schema"] = [
            {
                "name": "image_raw",
                "type": "FixedLen",
                "shape": [],
                "dtype": "str"
            },
            {
                "name": "label",
                "type": "FixedLen",
                "shape": [],
                "dtype": "int"
            }
        ]
    
- Name of image feature within schema dictionary.
######
        dynamic_threshold_dict["image_feature_name"] = "image_raw"
    
- Encoding of image: raw, png, or jpeg.
######
        dynamic_threshold_dict["image_encoding"] = "raw"
    
- Height of predownscaled image if NOT using multiple resolution records.
######
        dynamic_threshold_dict["image_predownscaled_height"] = 32
    
- Width of predownscaled image if NOT using multiple resolution records.
######
        dynamic_threshold_dict["image_predownscaled_width"] = 32
    
- Depth of image, number of channels.
######
        dynamic_threshold_dict["image_depth"] = 3
    
- Name of label feature within schema dictionary.
######
        dynamic_threshold_dict["label_feature_name"] = "label"
    
- Number of examples in one epoch of dynamic threshold training set.
######
        dynamic_threshold_dict["train_dataset_length"] = 400
    
- Number of examples in dynamic threshold training batch.
######
        dynamic_threshold_dict["train_batch_size"] = 32
    
- Number of steps/batches to evaluate for dynamic threshold.
######
        dynamic_threshold_dict["eval_steps"] = 10
    
- Whether to autotune input function performance for dynamic threshold datasets.
######
        dynamic_threshold_dict["input_fn_autotune"] = True
    
- How many steps to train dynamic threshold before saving a checkpoint.
######
        dynamic_threshold_dict["save_checkpoints_steps"] = 10000
    
- Max number of dynamic threshold checkpoints to keep.
######
        dynamic_threshold_dict["keep_checkpoint_max"] = 10
    
- Max number of times training loop can be restarted.
######
        dynamic_threshold_dict["max_training_loop_restarts"] = 10

- Whether using supervised dynamic thresholding or unsupervised.
######
        dynamic_threshold_dict["use_supervised"] = False

- Beta value for supervised F-beta score.
######
        supervised_dict = dict()
        supervised_dict["f_score_beta"] = 0.05

- Whether using sample or population covariance for dynamic threshold.
######
        unsupervised_dict = dict()
        unsupervised_dict["use_sample_covariance"] = True

- Max standard deviations of Mahalanobis distance to flag as outlier.
######
        unsupervised_dict["max_mahalanobis_stddevs"] = 3.0

- Dictionary of training configs.
######
        training_dict = dict()

- GCS location to write checkpoints, loss logs, and export models.
######
        training_dict["output_dir"] = "trained_models/experiment_0"
    
- Version of TensorFlow.
######
        training_dict["tf_version"] = 2.3
    
- Whether to use graph mode or not (eager).
######
        training_dict["use_graph_mode"] = True

- Which distribution strategy to use, if any.
######
        training_dict["distribution_strategy"] = "Mirrored"
    
- Whether we subclass models or use Functional API.
######
        training_dict["subclass_models"] = True
    
- Whether performing training phase 1 or not.
######
        training_dict["train_reconstruction"] = True
    
- Whether performing training phase 2 or not.
######
        training_dict["train_error_distribution"] = True
    
- Whether performing training phase 3 or not.
######
        training_dict["train_dynamic_threshold"] = True

- Dictionary of export configs.
######
        export_dict = dict()

- Most recent export's growth index so that there are no repeat exports.
######
        export_dict["most_recent_export_growth_idx"] = -1
    
- Most recent export's epoch index so that there are no repeat exports.
######
        export_dict["most_recent_export_epoch_idx"] = -1
    
- Whether to export SavedModel every growth phase.
######
        export_dict["export_every_growth_phase"] = True
    
- Whether to export SavedModel every epoch.
######
        export_dict["export_every_epoch"] = True
    
- Whether to export all growth phases or just current.
######
        export_dict["export_all_growth_phases"] = True

- Using a random noise vector Z with shape (batch_size, generator_latent_size) for berg.
    - Whether to export Z.
######
        export_dict["export_Z"] = True
    
- Whether to export generated images, G(z).
######
        export_dict["export_generated_images"] = True
    
- Whether to export encoded generated logits, E(G(z)).
######
        export_dict["export_encoded_generated_logits"] = True
    
- Whether to export encoded generated images, G(E(G(z))).
######
        export_dict["export_encoded_generated_images"] = True
    
- Whether to export Z generated images, Gd(z).
######
        export_dict["export_Z_generated_images"] = True

- Using a query image with shape (batch_size, height, width, depth)
    - Whether to export query images.
######
        export_dict["export_query_images"] = True

- Berg encoded exports.
    - Whether to export encoded query logits, E(x).
######
        export_dict["export_query_encoded_logits"] = True
    
- Whether to export encoded query images, G(E(x)).
######
        export_dict["export_query_encoded_images"] = True

- GANomaly encoded exports.
    - Whether to export generator encoded query logits, Ge(x).
######
        export_dict["export_query_gen_encoded_logits"] = True
    
- Whether to export generator encoded query images, G(x) = Gd(Ge(x)).
######
        export_dict["export_query_gen_encoded_images"] = True
    
- Whether to export encoder encoded query logits, E(G(x)).
######
        export_dict["export_query_enc_encoded_logits"] = True
    
- Whether to export encoder encoded query images, Gd(E(G(x))).
######
        export_dict["export_query_enc_encoded_images"] = True

- Anomaly exports.
    - Whether to export query anomaly images using sigmoid scaling.
######
        export_dict["export_query_anomaly_images_sigmoid"] = True
    
- Whether to export query anomaly images using linear scaling.
######
        export_dict["export_query_anomaly_images_linear"] = True
    
- Whether to export query Mahalanobis distances.
######
        export_dict["export_query_mahalanobis_distances"] = True
    
- Whether to export query Mahalanobis distance images using sigmoid scaling.
######
        export_dict["export_query_mahalanobis_distance_images_sigmoid"] = True
    
- Whether to export query Mahalanobis distance images using linear scaling.
######
        export_dict["export_query_mahalanobis_distance_images_linear"] = True
    
- Whether to export query pixel anomaly flag binary images.
######
        export_dict["export_query_pixel_anomaly_flag_images"] = True
    
- Whether to export query pixel anomaly flag binary images.
######
        export_dict["export_query_pixel_anomaly_flag_counts"] = True
    
- Whether to export query pixel anomaly flag binary images.
######
        export_dict["export_query_pixel_anomaly_flag_percentages"] = True
    
- Whether to export query anomaly scores, only for Berg.
######
        export_dict["export_query_anomaly_scores"] = False
    
- Whether to export query anomaly flags, only for Berg.
######
        export_dict["export_query_anomaly_flags"] = False

- Anomaly parameters.
    - The threshold value at which above flags scores images as anomalous.
######
        export_dict["anomaly_threshold"] = 5.0
    
- The anomaly convex combination factor for weighting the two anomaly losses.
######
        export_dict["anom_convex_combo_factor"] = 0.05

- Whether to print model summaries.
######
        export_dict["print_serving_model_summaries"] = False

- Dictionary of training default config.
######
        arguments = dict()

- training default config
######
        arguments["generator"] = get_generator_config()
        arguments["encoder"] = get_encoder_config()
        arguments["discriminator"] = get_discriminator_config()
        arguments["training"] = get_training_config()
        arguments["export"] = get_export_config()

- Full lists for full 1024x1024 network growth.
######
        full_conv_num_filters = [[512, 512], [512, 512], [512, 512], [512, 512], [256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]
        full_conv_kernel_sizes = [[4, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        full_conv_strides = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

- Set final image size as a multiple of 2, starting at 4.
######
        image_size = 1024
        num_conv_blocks = max(
            min(int(math.log(image_size, 2) - 1), len(full_conv_num_filters)), 1
        )
        arguments["conv_num_filters"] = full_conv_num_filters[0:num_conv_blocks]
        arguments["conv_kernel_sizes"] = full_conv_kernel_sizes[0:num_conv_blocks]
        arguments["conv_strides"] = full_conv_strides[0:num_conv_blocks]

- Get conv layer properties for generator and discriminator.
######
        (generator,
         discriminator) = (
            gan_layer_architecture_shapes.calc_generator_discriminator_conv_layer_properties(
                arguments["conv_num_filters"],
                arguments["conv_kernel_sizes"],
                arguments["conv_strides"],
                arguments["training"]["reconstruction"]["image_depth"]
            )
        )

- Split up generator properties into separate lists.
######
        (generator_base_conv_blocks,
         generator_growth_conv_blocks,
         generator_to_rgb_layers) = (
            gan_layer_architecture_shapes.split_up_generator_conv_layer_properties(
                generator,
                arguments["conv_num_filters"],
                arguments["conv_strides"],
                arguments["training"]["reconstruction"]["image_depth"]
            )
        )

- Generator list of list of lists of base conv block layer shapes.
######
        arguments["generator"]["base_conv_blocks"] = generator_base_conv_blocks

- Generator list of list of lists of growth conv block layer shapes.
######
        arguments["generator"]["growth_conv_blocks"] = generator_growth_conv_blocks
    
- Generator list of list of lists of to_RGB layer shapes.
######
        arguments["generator"]["to_rgb_layers"] = generator_to_rgb_layers

- Split up discriminator properties into separate lists.
######
        (discriminator_from_rgb_layers,
         discriminator_base_conv_blocks,
         discriminator_growth_conv_blocks) = (
            gan_layer_architecture_shapes.split_up_discriminator_conv_layer_properties(
                discriminator,
                arguments["conv_num_filters"],
                arguments["conv_strides"],
                arguments["training"]["reconstruction"]["image_depth"]
            )
        )

- Discriminator list of list of lists of from_RGB layer shapes.
######
        arguments["discriminator"]["from_rgb_layers"] = discriminator_from_rgb_layers
    
- Discriminator list of list of lists of base conv block layer shapes.
######
        arguments["discriminator"]["base_conv_blocks"] = discriminator_base_conv_blocks
    
- Discriminator list of list of lists of growth conv block layer shapes.
######
        arguments["discriminator"]["growth_conv_blocks"] = discriminator_growth_conv_blocks

- Image mask block pixel sizes list of lists.
######
        arguments["generator"]["image_mask_block_sizes"] = (
            image_masks.calculate_image_mask_block_sizes_per_resolution(
                num_resolutions=num_conv_blocks,
                min_height=arguments["generator"]["projection_dims"][0],
                min_width=arguments["generator"]["projection_dims"][1],
                pixel_mask_percent=(
                    arguments["generator"]["mask_generator_input_images_percent"]
                )
           )
        )
