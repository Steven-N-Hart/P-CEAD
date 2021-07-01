Implementation of [Unsupervised Learning of Anomaly Detection from Contaminated Image Data using Simultaneous Encoder Training](https://arxiv.org/abs/1905.11034).

# General Instruction

Checking [documentation folder](documentation) for instructions on how to train and evaluate the model either on Mayo Clinic Internal Dataset (if have access) or custom datasets. 
- [training configs explanation](documentation/training_defaults.md)
- [training on custom datasets either on GCP or locally](documentation/training_on_custom_datasets.md)
- [inference configs explanation](documentation/inference_defaults.md)
- [polygon confusion matrices configs explanation](documentation/confusion_matrice_defaults.md)
- [inference/prediction on custom datasets](documentation/prediction_local_on_custom_datasets.md)
- [examine loss logs](documentation/examine_loss_logs.md)

## Important Note
- If your institution does not have access to Google Cloud Services, please follow the instructions in below:
  - Follow the instructions on [training on custom datasets readme file](documentation/training_on_custom_datasets.md) and start to train the model on your 
  computational environment using the [train local notebook](notebooks/train_local.ipynb).
  - Follow the instructions on [predictions on custom datasets readme file](documentation/prediction_local_on_custom_datasets.md) and start to evaluate the 
  performance of the trained model on your computational environment using the [predict local notebook](notebooks/predict_local.ipynb).
- Based on our research findings, "GANomaly" is the ideal model architecture for the anomaly detection purpose. However, "Berg" architecture 
does not perform well in this study. 
  - If you are interested in training on Berg's architecture and prefer to evaluate its performance, please 
  follow the instruction on [predictions on custom datasets readme file](documentation/prediction_local_on_custom_datasets.md) and to evaluate the 
  performance of the trained model with "Berg" architecture using the [predict local notebook](notebooks/predict_local.ipynb).
  - This study has investigated the impacts of multiple combinations of loss terms on model optimizations, if you would examine the loss values of your 
  trained model, please follow the instructions on [examine loss logs](documentation/examine_loss_logs.md) file to evaluate your loss values using the
  [loss logs notebook](notebooks/loss_logs.ipynb)

# Training

## Files

In proganomaly_modules:
-   training_module: This directory contains the training code and package utilities. It can be used to train either locally or on GCP.

In proganomaly_modules/training_module:
-   trainer: This directory contains the training code python files.

In proganomaly_modules/training_module/trainer:
-   [anomaly.py](proganomaly_modules/training_module/trainer/anomaly.py): This python file contains the code for doing anomaly detection and anomaly localization.
-   [batch_calculate_distribution_statistics.py](proganomaly_modules/training_module/trainer/batch_calculate_distribution_statistics.py): This python file contains the code that calculates the parameters of a multivariate Gaussian distribution one batch of data at a time.
-   [checkpoints.py](proganomaly_modules/training_module/trainer/checkpoints.py): This python file contains the code for setting up what goes into a checkpoint, where they're written, etc.
-   [cli_parser.py](proganomaly_modules/training_module/trainer/cli_parser.py): This python file contains the code for parsing the user passed command-line arguments.
-   [custom_layers.py](proganomaly_modules/training_module/trainer/custom_layers.py): This python file contains the code for custom layers such as weight scaled dense and convolutional layers, pixel normalization, weighted summation, and minibatch stddev.
-   [datasets.py](proganomaly_modules/training_module/trainer/datasets.py): This python file contains the code for creating the multiple datasets used for the different training phases.
-   [defaults.py](proganomaly_modules/training_module/trainer/defaults.py): This python file contains the code for getting the default config values if JSON config is not provided.
-   [discriminators.py](proganomaly_modules/training_module/trainer/discriminators.py): This python file contains the code for building the network architectures for both discriminators and encoders.
-   [export_berg.py](proganomaly_modules/training_module/trainer/export_berg.py): This python file contains the code for exporting serving outputs to SavedModels for Berg paper.
-   [export_ganomaly.py](proganomaly_modules/training_module/trainer/export_ganomaly.py): This python file contains the code for exporting serving outputs to SavedModels for GANomaly paper.
-   [export.py](proganomaly_modules/training_module/trainer/export.py): This python file contains the code for exporting serving outputs to SavedModels.
-   [gan_layer_architecture_shapes.py](proganomaly_modules/training_module/trainer/gan_layer_architecture_shapes.py): This python file contains the code for creating the GAN layer architecture shapes.
-   [generators.py](proganomaly_modules/training_module/trainer/generators.py): This python file contains the code for building the network architecture for generators.
-   [image_masks.py](proganomaly_modules/training_module/trainer/image_masks.py): This python file contains the code for creating image masks for generator inputs.
-   [image_to_vector_networks.py](proganomaly_modules/training_module/trainer/image_to_vector_networks.py): This python file contains the code for building the network architectures for networks that have image inputs and vector outputs.
-   [instantiate_model.py](proganomaly_modules/training_module/trainer/instantiate_model.py): This python file contains the code for instantiating the model variables (layers, optimizers, etc.).
-   [logs.py](proganomaly_modules/training_module/trainer/logs.py): This python file contains the code for writing training logs to disk.
-   [losses_berg.py](proganomaly_modules/training_module/trainer/losses_berg.py): This python file contains the code for getting the losses for training and evaluation phases for Berg paper.
-   [losses_ganomaly.py](proganomaly_modules/training_module/trainer/losses_ganomaly.py): This python file contains the code for getting the losses for training and evaluation phases for GANomaly paper.
-   [losses.py](proganomaly_modules/training_module/trainer/losses.py): This python file contains the code for getting the losses for training and evaluation phases.
-   [model.py](proganomaly_modules/training_module/trainer/model.py): This python file contains the code for instantiating the main model class and calling the model training block.
-   [subclassed_models.py](proganomaly_modules/training_module/trainer/subclassed_models.py): This python file contains the code for subclassing our models rather than using the Functional API.
-   [task.py](proganomaly_modules/training_module/trainer/task.py): This python file contains the code for the entrypoint from the command-line where arguments are parsed and the model is called.
-   [train_dynamic_threshold.py](proganomaly_modules/training_module/trainer/train_dynamic_threshold.py): This python file contains the code training the dynamic threshold phase.
-   [train_error_distribution.py](proganomaly_modules/training_module/trainer/train_error_distribution.py): This python file contains the code for training the error distribution phase.
-   [train_post_reconstruction.py](proganomaly_modules/training_module/trainer/train_post_reconstruction.py): This python file contains the code for training the post-reconstruction error distribution and dynamic threshold phases.
-   [train_reconstruction.py](proganomaly_modules/training_module/trainer/train_reconstruction.py): This python file contains the code for the reconstruction phase using the network losses and variables to find the gradients and perform the variable update.
-   [train_step_dynamic_threshold.py](proganomaly_modules/training_module/trainer/train_step_dynamic_threshold.py): This python file contains the code for performing one train step during dynamic threshold training phase.
-   [train_step_error_distribution.py](proganomaly_modules/training_module/trainer/train_step_error_distribution.py): This python file contains the code for performing one train step during error distribution training phase.
-   [train_step_reconstruction.py](proganomaly_modules/training_module/trainer/train_step_reconstruction.py): This python file contains the code for performing one train step during reconstruction training phase.
-   [train_step.py](proganomaly_modules/training_module/trainer/train_step.py): This python file contains the code main class for performing one train step for all of the training phases.
-   [train.py](proganomaly_modules/training_module/trainer/train.py): This python file contains the code for main training class for updating the trained learned parameters for each of the training phases.
-   [training_inputs.py](proganomaly_modules/training_module/trainer/training_inputs.py): This python file contains the code for reading the TF Record input data files for training and evaluation phases.
-   [training_loop_dynamic_threshold.py](proganomaly_modules/training_module/trainer/training_loop_dynamic_threshold.py): This python file contains the code for running the dynamic threshold training loop.
-   [training_loop_error_distribution.py](proganomaly_modules/training_module/trainer/training_loop_error_distribution.py): This python file contains the code for running the error distribution training loop.
-   [training_loop_reconstruction.py](proganomaly_modules/training_module/trainer/training_loop_reconstruction.py): This python file contains the code for running the reconstruction training loop.
-   [training_loop.py](proganomaly_modules/training_module/trainer/training_loop.py): This python file contains the code main class for running the training loop for all training phases.
-   [vector_to_image_networks.py](proganomaly_modules/training_module/trainer/vector_to_image_networks.py): This python file contains the code for building the network architectures for networks that have vector inputs and image outputs.
    
An example training call stack would be:
1. Entrypoint from command-line ([task.py](proganomaly_modules/training_module/trainer/task.py)).
1. Parse command-line arguments ([cli_parser.py](proganomaly_modules/training_module/trainer/cli_parser.py)).
1. Gets the default config values if JSON config is not provided. ([defaults.py](proganomaly_modules/training_module/trainer/defaults.py)).
1. Calculate GAN layer architecture shapes ([gan_layer_architecture_shapes.py](proganomaly_modules/training_module/trainer/gan_layer_architecture_shapes.py)).
1. Call model ([task.py](proganomaly_modules/training_module/trainer/task.py)).
1. Instantiates main class ([model.py](proganomaly_modules/training_module/trainer/model.py)).
1. Creates input datasets ([training_inputs.py](proganomaly_modules/training_module/trainer/training_inputs.py)).
1. Instantiates models and optimizers ([instantiate_model.py](proganomaly_modules/training_module/trainer/instantiate_model.py)).
1. During model instantiation, generators are created ([generators.py](proganomaly_modules/training_module/trainer/generators.py)).
1. Depending on the type of generator, image-to-vector ([image_to_vector_networks.py](proganomaly_modules/training_module/trainer/image_to_vector_networks.py)) and vector-to-image ([vector_to_image_networks.py](proganomaly_modules/training_module/trainer/vector_to_image_networks.py)) networks are created.
1. During model instantiation, discriminators and encoders are created ([discriminators.py](proganomaly_modules/training_module/trainer/discriminators.py)).
1. Discriminators and encoders have image-to-vector ([image_to_vector_networks.py](proganomaly_modules/training_module/trainer/image_to_vector_networks.py)) networks created.
1. During model instantiation, some custom layers may be used ([custom_layers.py](proganomaly_modules/training_module/trainer/custom_layers.py)).
1. During model instantiation, distribution statistic calculators may be created for use in the error distribution and/or dynamic threshold training phases ([batch_calculate_distribution_statistics.py](proganomaly_modules/training_module/trainer/batch_calculate_distribution_statistics.py)).
1. Creates checkpoint machinery and possibly loads one ([checkpoints.py](proganomaly_modules/training_module/trainer/checkpoints.py)).

1. Begins reconstruction training loop, iterating over growths and inside that over epochs ([training_loop_reconstruction.py](proganomaly_modules/training_module/trainer/training_loop_reconstruction.py)).
1. Performs a reconstruction training step iteration ([train_step_reconstruction.py](proganomaly_modules/training_module/trainer/train_step_reconstruction.py)).
1. Train reconstruction networks for current train step ([train_reconstruction.py](proganomaly_modules/training_module/trainer/train_reconstruction.py)).
1. Get losses for reconstruction training phase ([losses.py](proganomaly_modules/training_module/trainer/losses.py)).
1. If berg architecture, get losses for reconstruction training phase ([losses_berg.py](proganomaly_modules/training_module/trainer/losses_berg.py)).
1. If GANomaly architecture, get losses for reconstruction training phase ([losses_ganomaly.py](proganomaly_modules/training_module/trainer/losses_ganomaly.py)).
1. If GANomaly architecture, possibly masks generator input images ([image_masks.py](proganomaly_modules/training_module/trainer/image_masks.py)).
1. For reconstruction training, calculate gradients and update weights ([train_reconstruction.py](proganomaly_modules/training_module/trainer/train_reconstruction.py)).
1. Every so often or at the end of reconstruction training export SavedModel for serving ([export.py](proganomaly_modules/training_module/trainer/export.py)).
1. If berg architecture, export for reconstruction training phase ([export_berg.py](proganomaly_modules/training_module/trainer/export_berg.py)).
1. If GANomaly architecture, export for reconstruction training phase ([export_ganomaly.py](proganomaly_modules/training_module/trainer/export_ganomaly.py)).
1. Gets used if user wants to serve anomaly detection and/or anomaly localization([anomaly.py](proganomaly_modules/training_module/trainer/anomaly.py)).
1. Returns to main class to start error distribution training phase ([model.py](proganomaly_modules/training_module/trainer/model.py)).

1. Begins error distribution training loop ([training_loop_error_distribution.py](proganomaly_modules/training_module/trainer/training_loop_error_distribution.py)).
1. Performs an error distribution training step iteration ([train_step_error_distribution.py](proganomaly_modules/training_module/trainer/train_step_error_distribution.py)).
1. Train error distribution statistic parameters for current train step ([train_error_distribution.py](proganomaly_modules/training_module/trainer/train_error_distribution.py)).
1. Every so often or at the end of error distribution training export SavedModel for serving ([export.py](proganomaly_modules/training_module/trainer/export.py)).
1. If berg architecture, export for reconstruction training phase ([export_berg.py](proganomaly_modules/training_module/trainer/export_berg.py)).
1. If GANomaly architecture, export for reconstruction training phase ([export_ganomaly.py](proganomaly_modules/training_module/trainer/export_ganomaly.py)).
1. Calculates inverse covariance matrix for error distribution ([training_loop_error_distribution.py](proganomaly_modules/training_module/trainer/training_loop_error_distribution.py)).
1. Returns to main class to start dynamic threshold training phase ([model.py](proganomaly_modules/training_module/trainer/model.py)).

1. Begins dynamic threshold training loop ([training_loop_dynamic_threshold.py](proganomaly_modules/training_module/trainer/training_loop_dynamic_threshold.py)).
1. Performs a dynamic threshold training step iteration ([train_step_dynamic_threshold.py](proganomaly_modules/training_module/trainer/train_step_dynamic_threshold.py)).
1. Train dynamic threshold statistic parameters for current train step ([train_dynamic_threshold.py](proganomaly_modules/training_module/trainer/train_dynamic_threshold.py)).
1. Every so often or at the end of error distribution training export SavedModel for serving ([export.py](proganomaly_modules/training_module/trainer/export.py)).
1. If berg architecture, export for reconstruction training phase ([export_berg.py](proganomaly_modules/training_module/trainer/export_berg.py)).
1. If GANomaly architecture, export for reconstruction training phase ([export_ganomaly.py](proganomaly_modules/training_module/trainer/export_ganomaly.py)).
1. Calculates dynamic threshold from learned Mahalanobis distance distribution statistic parameters ([training_loop_dynamic_threshold.py](proganomaly_modules/training_module/trainer/training_loop_dynamic_threshold.py)).


## Perform training

Example JSON config to train model (defaults.py contains schema information):
```json
{
  "generator":{
    "architecture":"GANomaly",
    "train":true,
    "train_steps":1,
    "normalize_latents":true,
    "use_pixel_norm":true,
    "pixel_norm_epsilon":1e-08,
    "projection_dims":[
      4,
      4,
      512
    ],
    "leaky_relu_alpha":0.2,
    "final_activation":"None",
    "l1_regularization_scale":0.0,
    "l2_regularization_scale":0.0,
    "optimizer":"Adam",
    "learning_rate":0.001,
    "adam_beta1":0.0,
    "adam_beta2":0.99,
    "adam_epsilon":1e-08,
    "clip_gradients":null,
    "berg":{
      
    },
    "GANomaly":{
      "use_unet_skip_connections":[
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "mask_generator_input_images_percent":0.2,
      "image_mask_block_random_shift_amount":0,
      "use_shuffle_image_masks":true,
      "add_uniform_noise_to_z":true,
      "add_uniform_noise_to_fake_images":true,
      "image_mask_block_sizes":[
        [
          1,
          1,
          1
        ],
        [
          6,
          3,
          1,
          1,
          1
        ],
        [
          25,
          13,
          6,
          3,
          2,
          1,
          1
        ],
        [
          102,
          51,
          25,
          13,
          6,
          3,
          2,
          1,
          1
        ],
        [
          409,
          205,
          102,
          51,
          26,
          13,
          6,
          3,
          2,
          1,
          1
        ],
        [
          1638,
          819,
          409,
          205,
          102,
          51,
          26,
          13,
          6,
          3,
          2,
          1,
          1
        ],
        [
          6553,
          3277,
          1638,
          819,
          410,
          205,
          102,
          51,
          26,
          13,
          6,
          3,
          2,
          1,
          1
        ],
        [
          26214,
          13107,
          6553,
          3277,
          1638,
          819,
          410,
          205,
          102,
          51,
          26,
          13,
          6,
          3,
          2,
          1,
          1
        ],
        [
          104857,
          52429,
          26214,
          13107,
          6554,
          3277,
          1638,
          819,
          410,
          205,
          102,
          51,
          26,
          13,
          6,
          3,
          2,
          1,
          1
        ]
      ]
    },
    "losses":{
      "berg":{
        "D_of_G_of_z_loss_weight":0.0,
        "D_of_G_of_E_of_x_loss_weight":0.0,
        "D_of_G_of_E_of_G_of_z_loss_weight":0.0,
        "z_minus_E_of_G_of_z_l1_loss_weight":0.0,
        "z_minus_E_of_G_of_z_l2_loss_weight":0.0,
        "G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight":0.0,
        "G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight":0.0,
        "E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight":0.0,
        "E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight":0.0,
        "x_minus_G_of_E_of_x_l1_loss_weight":0.0,
        "x_minus_G_of_E_of_x_l2_loss_weight":0.0
      },
      "GANomaly":{
        "D_of_G_of_x_loss_weight":1.0,
        "x_minus_G_of_x_l1_loss_weight":0.0,
        "x_minus_G_of_x_l2_loss_weight":1.0,
        "Ge_of_x_minus_E_of_G_of_x_l1_loss_weight":0.0,
        "Ge_of_x_minus_E_of_G_of_x_l2_loss_weight":0.0
      }
    },
    "base_conv_blocks":[
      [
        [
          4,
          4,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ]
    ],
    "growth_conv_blocks":[
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          512,
          256,
          1,
          1
        ],
        [
          3,
          3,
          256,
          256,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          256,
          128,
          1,
          1
        ],
        [
          3,
          3,
          128,
          128,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          128,
          64,
          1,
          1
        ],
        [
          3,
          3,
          64,
          64,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          64,
          32,
          1,
          1
        ],
        [
          3,
          3,
          32,
          32,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          32,
          16,
          1,
          1
        ],
        [
          3,
          3,
          16,
          16,
          1,
          1
        ]
      ]
    ],
    "to_rgb_layers":[
      [
        [
          1,
          1,
          512,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          512,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          512,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          512,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          256,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          128,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          64,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          32,
          3,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          16,
          3,
          1,
          1
        ]
      ]
    ]
  },
  "encoder":{
    "create":false,
    "train":false,
    "use_minibatch_stddev":true,
    "minibatch_stddev_group_size":4,
    "minibatch_stddev_use_averaging":true,
    "leaky_relu_alpha":0.2,
    "l1_regularization_scale":0.0,
    "l2_regularization_scale":0.0,
    "optimizer":"Adam",
    "learning_rate":0.001,
    "adam_beta1":0.0,
    "adam_beta2":0.99,
    "adam_epsilon":1e-08,
    "clip_gradients":null,
    "losses":{
      "berg":{
        "D_of_G_of_E_of_x_loss_weight":0.0,
        "D_of_G_of_E_of_G_of_z_loss_weight":0.0,
        "z_minus_E_of_G_of_z_l1_loss_weight":0.0,
        "z_minus_E_of_G_of_z_l2_loss_weight":0.0,
        "G_of_z_minus_G_of_E_of_G_of_z_l1_loss_weight":0.0,
        "G_of_z_minus_G_of_E_of_G_of_z_l2_loss_weight":0.0,
        "E_of_x_minus_E_of_G_of_E_of_x_l1_loss_weight":0.0,
        "E_of_x_minus_E_of_G_of_E_of_x_l2_loss_weight":0.0,
        "x_minus_G_of_E_of_x_l1_loss_weight":0.0,
        "x_minus_G_of_E_of_x_l2_loss_weight":0.0
      },
      "GANomaly":{
        "Ge_of_x_minus_E_of_G_of_x_l1_loss_weight":0.0,
        "Ge_of_x_minus_E_of_G_of_x_l2_loss_weight":1.0
      }
    }
  },
  "discriminator":{
    "create":true,
    "train":true,
    "train_steps":1,
    "use_minibatch_stddev":true,
    "minibatch_stddev_group_size":4,
    "minibatch_stddev_use_averaging":true,
    "leaky_relu_alpha":0.2,
    "l1_regularization_scale":0.0,
    "l2_regularization_scale":0.0,
    "optimizer":"Adam",
    "learning_rate":0.001,
    "adam_beta1":0.0,
    "adam_beta2":0.99,
    "adam_epsilon":1e-08,
    "clip_gradients":null,
    "gradient_penalty_coefficient":10.0,
    "gradient_penalty_target":1.0,
    "epsilon_drift":0.001,
    "losses":{
      "D_of_x_loss_weight":1.0,
      "berg":{
        "D_of_G_of_z_loss_weight":0.0,
        "D_of_G_of_E_of_x_loss_weight":0.0,
        "D_of_G_of_E_of_G_of_z_loss_weight":0.0
      },
      "GANomaly":{
        "D_of_G_of_x_loss_weight":1.0
      }
    },
    "from_rgb_layers":[
      [
        [
          1,
          1,
          3,
          512,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          512,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          512,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          512,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          256,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          128,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          64,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          32,
          1,
          1
        ]
      ],
      [
        [
          1,
          1,
          3,
          16,
          1,
          1
        ]
      ]
    ],
    "base_conv_blocks":[
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          4,
          4,
          512,
          512,
          1,
          1
        ]
      ]
    ],
    "growth_conv_blocks":[
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          512,
          512,
          1,
          1
        ],
        [
          3,
          3,
          512,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          256,
          256,
          1,
          1
        ],
        [
          3,
          3,
          256,
          512,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          128,
          128,
          1,
          1
        ],
        [
          3,
          3,
          128,
          256,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          64,
          64,
          1,
          1
        ],
        [
          3,
          3,
          64,
          128,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          32,
          32,
          1,
          1
        ],
        [
          3,
          3,
          32,
          64,
          1,
          1
        ]
      ],
      [
        [
          3,
          3,
          16,
          16,
          1,
          1
        ],
        [
          3,
          3,
          16,
          32,
          1,
          1
        ]
      ]
    ]
  },
  "training":{
    "output_dir":"gs://my-bucket/trained_models/experiment",
    "tf_version":2.3,
    "use_graph_mode":true,
    "distribution_strategy":"Mirrored",
    "subclass_models":true,
    "train_reconstruction":true,
    "train_error_distribution":true,
    "train_dynamic_threshold":true,
    "reconstruction":{
      "use_multiple_resolution_records":true,
        "train_file_patterns":[
          "data/cifar10_car/train_4x4_*.tfrecord",
          "data/cifar10_car/train_8x8_*.tfrecord",
          "data/cifar10_car/train_16x16_*.tfrecord",
          "data/cifar10_car/train_32x32_*.tfrecord"
        ],
        "eval_file_patterns":[
          "data/cifar10_car/test_4x4_*.tfrecord",
          "data/cifar10_car/test_8x8_*.tfrecord",
          "data/cifar10_car/test_16x16_*.tfrecord",
          "data/cifar10_car/test_32x32_*.tfrecord"
        ],
      "dataset":"tf_record",
      "tf_record_example_schema":[
        {
          "name":"image/encoded",
          "type":"FixedLen",
          "shape":[
            
          ],
          "dtype":"str"
        },
        {
          "name":"image/name",
          "type":"FixedLen",
          "shape":[
            
          ],
          "dtype":"str"
        }
      ],
      "image_feature_name":"image/encoded",
      "image_encoding":"png",
      "image_predownscaled_height":1024,
      "image_predownscaled_width":1024,
      "image_depth":3,
      "label_feature_name":"",
      "num_epochs_schedule":[
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ],
      "train_dataset_length":330415,
      "train_batch_size_schedule":[
        32,
        16,
        16,
        16,
        16,
        4,
        2,
        2,
        1
      ],
      "eval_batch_size_schedule":[
        32,
        16,
        16,
        16,
        16,
        4,
        2,
        2,
        1
      ],
      "eval_steps":1,
      "num_examples_until_growth_schedule":[
        330415,
        330415,
        330415,
        330415,
        330415,
        330415,
        330415,
        330415,
        330415
      ],
      "num_steps_until_growth_schedule":[
        10325,
        20650,
        20650,
        20650,
        20650,
        82603,
        165207,
        165207,
        330415
      ],
      "input_fn_autotune":true,
      "log_step_count_steps":100,
      "save_summary_steps":100,
      "write_loss_summaries":false,
      "write_generator_image_summaries":false,
      "write_encoder_image_summaries":false,
      "write_variable_histogram_summaries":false,
      "write_gradient_histogram_summaries":false,
      "save_checkpoints_steps":10000,
      "keep_checkpoint_max":100,
      "checkpoint_every_growth_phase":true,
      "checkpoint_every_epoch":true,
      "checkpoint_growth_idx":0,
      "checkpoint_epoch_idx":0,
      "checkpoint_save_path":"",
      "store_loss_logs":true,
      "normalized_loss_logs":true,
      "print_training_model_summaries":false,
      "initial_growth_idx":0,
      "initial_epoch_idx":0,
      "max_training_loop_restarts":20,
      "use_equalized_learning_rate":true,
      "normalize_reconstruction_losses":true
    },
    "error_distribution":{
      "use_multiple_resolution_records":false,
      "train_file_pattern":"data/cifar10_car/train_32x32_*.tfrecord",
      "eval_file_pattern":"data/cifar10_car/train_32x32_*.tfrecord",
      "dataset":"tf_record",
      "tf_record_example_schema":[
        {
          "name":"image/encoded",
          "type":"FixedLen",
          "shape":[
            
          ],
          "dtype":"str"
        },
        {
          "name":"image/name",
          "type":"FixedLen",
          "shape":[
            
          ],
          "dtype":"str"
        }
      ],
      "image_feature_name":"image/encoded",
      "image_encoding":"png",
      "image_predownscaled_height":1024,
      "image_predownscaled_width":1024,
      "image_depth":3,
      "label_feature_name":"",
      "train_dataset_length":44693,
      "train_batch_size":16,
      "eval_steps":1,
      "input_fn_autotune":true,
      "save_checkpoints_steps":10000,
      "keep_checkpoint_max":100,
      "max_training_loop_restarts":20,
      "use_sample_covariance":true
    },
    "dynamic_threshold":{
      "use_multiple_resolution_records":false,
      "train_file_pattern":"data/cifar10_car/train_32x32_*.tfrecord",
      "eval_file_pattern":"data/cifar10_car/train_32x32_*.tfrecord",
      "dataset":"tf_record",
      "tf_record_example_schema":[
        {
          "name":"image/encoded",
          "type":"FixedLen",
          "shape":[
            
          ],
          "dtype":"str"
        },
        {
          "name":"image/name",
          "type":"FixedLen",
          "shape":[
            
          ],
          "dtype":"str"
        }
      ],
      "image_feature_name":"image/encoded",
      "image_encoding":"png",
      "image_predownscaled_height":1024,
      "image_predownscaled_width":1024,
      "image_depth":3,
      "label_feature_name":"",
      "train_dataset_length":52517,
      "train_batch_size":16,
      "eval_steps":1,
      "input_fn_autotune":true,
      "save_checkpoints_steps":10000,
      "keep_checkpoint_max":100,
      "max_training_loop_restarts":20,
      "use_supervised":false,
      "supervised":{
        "f_score_beta":0.05
      },
      "unsupervised":{
        "use_sample_covariance":true,
        "max_mahalanobis_stddevs":3.0
      }
    }
  },
  "export":{
    "most_recent_export_growth_idx":-1,
    "most_recent_export_epoch_idx":-1,
    "export_every_growth_phase":true,
    "export_every_epoch":true,
    "export_all_growth_phases":true,
    "export_Z":true,
    "export_generated_images":true,
    "export_encoded_generated_logits":true,
    "export_encoded_generated_images":true,
    "export_Z_generated_images":true,
    "export_query_images":true,
    "export_query_encoded_logits":true,
    "export_query_encoded_images":true,
    "export_query_gen_encoded_logits":true,
    "export_query_gen_encoded_images":true,
    "export_query_enc_encoded_logits":true,
    "export_query_enc_encoded_images":true,
    "export_query_anomaly_images_sigmoid":true,
    "export_query_anomaly_images_linear":true,
    "export_query_mahalanobis_distances":true,
    "export_query_mahalanobis_distance_images_sigmoid":true,
    "export_query_mahalanobis_distance_images_linear":true,
    "export_query_pixel_anomaly_flag_images":true,
    "export_query_pixel_anomaly_flag_counts":true,
    "export_query_pixel_anomaly_flag_percentages":true,
    "export_query_anomaly_scores":false,
    "export_query_anomaly_flags":false,
    "anomaly_threshold":5.0,
    "anom_convex_combo_factor":0.05,
    "print_serving_model_summaries":false
  },
  "conv_num_filters":[
    [
      512,
      512
    ],
    [
      512,
      512
    ],
    [
      512,
      512
    ],
    [
      512,
      512
    ],
    [
      256,
      256
    ],
    [
      128,
      128
    ],
    [
      64,
      64
    ],
    [
      32,
      32
    ],
    [
      16,
      16
    ]
  ],
  "conv_kernel_sizes":[
    [
      4,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ],
    [
      3,
      3
    ]
  ],
  "conv_strides":[
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ],
    [
      1,
      1
    ]
  ]
}
```

The simplest way to call a GCP CAIP distributed training job is to create experiment config JSON archetypes like above and provide the GCS filepath in the CLI.
If you want to make a quick change to a parameter without changing the JSON, then you can create a dictionary with the same nested structure and pass that to the command line as an override of whatever that parameter(s) is within the JSON.
```bash
gcloud ai-platform jobs submit training ${JOBNAME} \
    --region=${REGION} \
    --module-name=trainer.task \
    --package-path=$PWD/proganomaly_modules/training_module/trainer \
    --job-dir=${OUTPUT_DIR} \
    --staging-bucket=gs://${BUCKET} \
    --config=config.yaml \
    --runtime-version=${TFVERSION} \
    --python-version=${PYTHON_VERSION} \
    -- \
    --json_config_gcs_path=${JSON_CONFIG_GCS_PATH} \
    --json_overrides=${JSON_OVERRIDES}
```

where config.yaml is like the following example where the master and accelerators need to follow these [pairs](https://cloud.google.com/ai-platform/training/docs/using-gpus?hl=cs):
```yaml
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-highmem-16
  masterConfig:
    acceleratorConfig:
      count: 2
      type: NVIDIA_TESLA_V100
```

# Inference

## Files

In proganomaly_modules:
-   inference_module: This directory contains the inference code for getting predictions and visualization.

In proganomaly_modules/inference_module:
-   [get_predictions.py](proganomaly_modules/inference_module/get_predictions.py): This python file contains the code for getting either one growth's or all of the growths' predictions from an exported SavedModel.
-   [image_utils.py](proganomaly_modules/inference_module/image_utils.py): This python file contains the code for image processing and plotting of images.
-   [inference.py](proganomaly_modules/inference_module/inference.py): This python file contains the code for the entrypoint of the inference pipeline for plotting predictions from exported SavedModels.
-   [inference_inputs.py](proganomaly_modules/inference_module/inference_inputs.py): This python file contains the code for reading the TF Record input data files for inference.

An example inference call stack would be:
1. Get query_images within notebook ([inference_inputs.py](proganomaly_modules/inference_module/inference_inputs.py)).
1. Entrypoint from notebook for plotting predictions from exports ([inference.py](proganomaly_modules/inference_module/inference.py)).
1. Get predictions from currently iterated export ([get_predictions.py](proganomaly_modules/inference_module/get_predictions.py)).
1. Plot predictions ([image_utils.py](proganomaly_modules/inference_module/image_utils.py)).

## Perform inference
```python
generator_architecture = "GANomaly"

if generator_architecture == "berg":
    predictions_by_growth = inference.plot_all_exports_by_architecture(
        Z=Z,
        query_images=query_images,
        exports_on_gcs=True,
        export_start_idx=0,
        export_end_idx=17,
        max_size=1024,
        only_output_growth_set={i for i in range(9)},
        num_rows=1,
        generator_architecture="berg",
        overrides={
            "output_dir": "gs://.../trained_models/experiment_0",

            "output_generated_images": True,
            "output_encoded_generated_images": True,

            "output_query_images": True,

            "output_query_encoded_images": True,

            "output_query_anomaly_images_sigmoid": True
        }
    )
elif generator_architecture == "GANomaly":
    predictions_by_growth = inference.plot_all_exports_by_architecture(
        Z=Z,
        query_images=query_images,
        exports_on_gcs=False,
        export_start_idx=0,
        export_end_idx=17,
        max_size=1024,
        only_output_growth_set={i for i in range(9)},
        num_rows=1,
        generator_architecture="GANomaly",
        overrides={
            "output_dir": "gs://.../trained_models/experiment_0",

            "output_query_images": True,

            "output_query_gen_encoded_images": True,
            "output_query_enc_encoded_images": True,

            "output_query_anomaly_images_sigmoid": True
        }
    )
```

# Dataflow Stitch Pipeline

## Files

In proganomaly_modules:
-   beam_image_stitch: This directory contains the code for using Dataflow to perform distributed inference and then stitch results together.

In proganomaly_modules/beam_image_stitch:
-   components: This directory contains the code for all of the pipeline components.
-   [beam_image_stitch.py](proganomaly_modules/beam_image_stitch/beam_image_stitch.py): This python file contains the code for the Dataflow pipeline and parsing the config args from the command line.

In proganomaly_modules/beam_image_stitch/components:
-   [confusion_matrix.py](proganomaly_modules/beam_image_stitch/components/confusion_matrix.py): This python file contains the code for calculating the confusion matrix between the thresholded KDE grayscale image and the annotations image. No polygons.
-   [images.py](proganomaly_modules/beam_image_stitch/components/images.py): This python file contains the code for the Dataflow pipeline and parsing the config args from the command line.
-   [inference.py](proganomaly_modules/beam_image_stitch/components/inference.py): This python file contains the code for performing inference using the SavedModels.
-   [patch_coordinates.py](proganomaly_modules/beam_image_stitch/components/patch_coordinates.py): This python file contains the code for creating the patch coordinates list to use for bounding polygons.
-   [pre_inference_png.py](proganomaly_modules/beam_image_stitch/components/pre_inference_png.py): This python file contains the code for creating the image element grid to be used for inference using PNG images.
-   [pre_inference_wsi.py](proganomaly_modules/beam_image_stitch/components/pre_inference_wsi.py): This python file contains the code for creating the image element grid to be used for inference using WSI files.
-   [segmentation.py](proganomaly_modules/beam_image_stitch/components/segmentation.py): This python file contains the code for saving the segmentation model results.

# Dataflow Polygon Confusion Matrix Pipeline

## Files

In proganomaly_modules:
-   beam_polygon_confusion_matrix: This directory contains the code for using Dataflow to threshold KDE grayscale images, create dilated Shapely Polygons, and then calculate confusion matrix metrics when compared against the annotations.

In proganomaly_modules/beam_polygon_confusion_matrix:
-   components: This directory contains the code for all of the pipeline components.
-   [polygon_threshold_dilation_confusion_matrix.py](proganomaly_modules/beam_polygon_confusion_matrix/polygon_threshold_dilation_confusion_matrix.py): This python file contains the code for the Dataflow pipeline and parsing the config args from the command line.

In proganomaly_modules/beam_polygon_confusion_matrix/components:
-   [confusion_matrix.py](proganomaly_modules/beam_polygon_confusion_matrix/components/confusion_matrix.py): This python file contains the code for saving the confusion matrix dictionary.
-   [polygon.py](proganomaly_modules/beam_polygon_confusion_matrix/components/polygon.py): This python file contains the code for generating thresholded and dilated polygons.
-   [pre_polygon.py](proganomaly_modules/beam_polygon_confusion_matrix/components/pre_polygon.py): This python file contains the code for generating the cartesian product of thresholds and dilation factors.