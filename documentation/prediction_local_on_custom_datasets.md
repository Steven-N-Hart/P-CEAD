# Proganomaly Predictions(/Inference) Local on Custom Dataset

## License

Copyright 2020 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at [Apache License Page](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
===================================================

## Example Dataset Introduction
This documentation intends to provide user the introduction on running 
[prediction (/inference) code locally](notebooks/predict_local.ipynb) on the 
user custom dataset. We take the ICIAR 2018 Grand Challenge on BreAst Cancer Histology images 
[BACH Dataset](https://iciar2018-challenge.grand-challenge.org/Dataset/).

We used the microscopy histology images in "tif" format from this Dataset for evaluation purposes. We 
extracted image patches with the patch size be 512x512 from each of the images from this dataset. Just a quick note, the 
data we used in our publication, the image patch size is 1024x1024. The reason we add this documentation intends to 
provide user the insight on how to evaluate the models on a different image size locally. 

Since we have already provide a explanation for all prediction (/inference) input arguments in 
[inference_defaults.md](documentation/inference_defaults.md), instead of listing all training configs for training on the 
BACH dataset, we only provide the input training arguments that are necessary to change in order to start the training 
job on the BACH dataset. 

## Required Updated Prediction (/Inference) Local Arguments 

Get query images

        dataset_name = "bach_breast"
        
        elif dataset_name == "bach_breast:
            size = 512
            block_idx = int(math.log(size, 2)) - 2
            dataset = inference_inputs.read_dataset(
                file_pattern="gs://path to inference image files in tfrecords format".format(
                    "image name", 8 - block_idx
                ),
                batch_size=batch_size_query_images,
                block_idx=block_idx,
                params={
                    "use_multiple_resolution_records": True,
                    "tf_record_example_schema": [
                        {
                            "name": "image/encoded",
                            "type": "FixedLen",
                            "shape": [],
                            "dtype": "str"
                        },
                        {
                            "name": "image/name",
                            "type": "FixedLen",
                            "shape": [],
                            "dtype": "str"
                        },
                        {
                            "name": "image/width",
                            "type": "FixedLen",
                            "shape": [],
                            "dtype": "int"
                        },
                        {
                            "name": "image/height",
                            "type": "FixedLen",
                            "shape": [],
                            "dtype": "int"
                        }
                    ],
                    "image_feature_name": "image/encoded",
                    "image_encoding": "png",
                    "image_predownscaled_height": 512,
                    "image_predownscaled_width": 512,
                    "image_depth": 3,
                    "label_feature_name": "",
                    "input_fn_autotune": False,
                    "generator_projection_dims": [4, 4, 512]
                }
            )().take(1)
        
            for batch in dataset:
                numpy_batch = {k: v.numpy() for k, v in batch.items()}

Plot exports
- Berg architecture model
######
        predictions_by_growth = gan_inference.plot_all_exports_by_architecture(
            Z=Z,
            query_images=query_images,
            exports_on_gcs=False,
            export_start_idx=0,
            export_end_idx=1,
            max_size=512,
            only_output_growth_set={i for i in range(9)},
            num_rows=1,
            generator_architecture="berg",
            overrides={
                # use gcs path when exports_on_gcs be True
                "output_dir": "/.../train_berg_models",

                "export_all_growth_phases": False,

                "output_generated_images": True,
                "output_encoded_generated_images": True,

                "output_query_images": True,

                "output_query_encoded_images": True,

                "output_query_anomaly_images_sigmoid": True,
                "output_query_anomaly_images_linear": True,

                "output_query_mahalanobis_distances": True,
                "output_query_mahalanobis_distance_images_sigmoid": True,
                "output_query_mahalanobis_distance_images_linear": True,

                "output_query_pixel_anomaly_flags": True,

                "output_query_anomaly_scores": True,
                "output_query_anomaly_flags": True
            }
        )
        
- GANomaly architecture model 
######                
        predictions_by_growth = gan_inference.plot_all_exports_by_architecture(
            Z=None,
            query_images=query_images,
            exports_on_gcs=False,
            export_start_idx=0,
            export_end_idx=1,
            max_size=512,
            only_output_growth_set={i for i in range(9)},
            num_rows=1,
            generator_architecture="GANomaly",
            overrides={
                # use gcs path when exports_on_gcs be True
                "output_dir": "/.../trained_GANomaly_models",
                
                "export_all_growth_phases": False,

                "output_generated_images": True,
                "output_encoded_generated_images": True,

                "output_query_images": True,

                "output_query_encoded_images": True,

                "output_query_anomaly_images_sigmoid": True,
                "output_query_anomaly_images_linear": True,

                "output_query_mahalanobis_distances": True,
                "output_query_mahalanobis_distance_images_sigmoid": True,
                "output_query_mahalanobis_distance_images_linear": True,

                "output_query_pixel_anomaly_flags": True,

                "output_query_anomaly_scores": True,
                "output_query_anomaly_flags": True
            }
        )

- We set exports_on_gcs be False, which means we copied the trained model outputs locally rather than call it from 
gcs path remotely. Therefore, we have to do the following bash commands,
######
        gsutil cp -r gs://.../trained_model /.../trained_models/export

