# Proganomaly Training on Custom Dataset

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
This documentation intends to provide user the introduction on 
[GCP training](notebooks/mayo_train_gcp.ipynb) or 
[Local training](notebooks/train_local.ipynb)
on the user custom dataset. We take the ICIAR 
2018 Grand Challenge on BreAst Cancer Histology images 
[BACH Dataset](https://iciar2018-challenge.grand-challenge.org/Dataset/).

We used the microscopy histology images in "tif" format from this Dataset for training and evaluation purposes. We 
extracted image patches with the patch size be 512x512 from each of the images from this dataset. Just a quick note, the 
data we used in our publication, the image patch size is 1024x1024. The reason we add this documentation intends to 
provide user the insight on how to train and evaluate the models on a different image size. 

Since we have already provide a explanation for all training input arguments in 
[training_defaults.md](documentation/training_defaults.md), instead of listing all training configs for training on the 
BACH dataset, we only provide the input training arguments that are necessary to change in order to start the training 
job on the BACH dataset. 

## Required Updated Training Arguments 

Reconstruction training parameters.

- GCS locations to read reconstruction training data.
######
        reconstruction_dict["train_file_patterns"] = [
            tf.io.gfile.glob(
                pattern=gs://.../BACH/train/*/*_L{}.tfrecords".format(i)
            )[0:150]
            for i in range(9 - 1, 0, -1)
        ]
        
- GCS locations to read reconstruction evaluation data.
######
        reconstruction_dict["eval_file_patterns"] = [
            tf.io.gfile.glob(
                pattern=gs://.../BACH/train/*/*_L{}.tfrecords".format(i)
            )[0:150]
            for i in range(9 - 1, 0, -1)
        ]
        
- Height of predownscaled image if NOT using multiple resolution records.
######
        reconstruction_dict["image_predownscaled_height"] = 1024
        
- Width of predownscaled image if NOT using multiple resolution records.
######
        reconstruction_dict["image_predownscaled_width"] = 1024

- Number of examples in one epoch of reconstruction training set.
    - "train_dataset_length" = (#slides * #image_patches per slide // batch_size * #gpu) * (batch_size * #gpu)
    - In BACH dataset, for each slide, we have 48 image patches with patch size be 512x512. 
    - We have 150 slides from the BACH dataset for training, the batch size is 32, number of gpu is 2. Therefore, 
    the train_dataset_length be 7168 in below is calculated by
    
        (150 * 48) // (32 * 2) * (32 * 2) = 7200 // 64 * 64 = 112 * 64 = 7168
    - In order for user to fully understand this, we provide another example with 25 slides with the rest values be 
    the same. Therefore, the train_dataset_length will be 1152, which is calculated by
    
        (25 * 48) // (32 * 2) * (32 * 2) = 1200 // 64 * 64 = 18 * 64 = 1152
    
    Since we know 1200 / 64 = 18.75, and 1200 // 64 = 18.
######
        reconstruction_dict["train_dataset_length"] = 7168

Error distribution training parameters.

- GCS locations to read error distribution training data.
######
        error_distribution_dict["train_file_pattern"] = tf.io.gfile.glob(
            pattern="gs://.../BACH/train/*/*_L{}.tfrecords".format(1)
        )[150:175]
        
- GCS locations to read error distribution training data.
######
        error_distribution_dict["eval_file_pattern"] = tf.io.gfile.glob(
            pattern="gs://.../BACH/train/*/*_L{}.tfrecords".format(1)
        )[150:170]

- Height of predownscaled image if NOT using multiple resolution records.
######
        error_distribution_dict["image_predownscaled_height"] = 1024
        
- Width of predownscaled image if NOT using multiple resolution records.
######
        error_distribution_dict["image_predownscaled_width"] = 1024

- Number of examples in one epoch of error distribution training set.
    - "train_dataset_length" = (#slides * #image_patches per slide // batch_size * #gpu) * (batch_size * #gpu)
######
        error_distribution_dict["train_dataset_length"] = 960

Dynamic threshold training parameters.

- GCS locations to read dynamic threshold training data.
######
        dynamic_threshold_dict["train_file_pattern"] = tf.io.gfile.glob(
            pattern="gs://.../BACH/train/*/*_L{}.tfrecords".format(1)
        )[170:200]
- GCS locations to read dynamic threshold evaluation data.
######
        dynamic_threshold_dict["eval_file_pattern"] = tf.io.gfile.glob(
            pattern="gs://.../BACH/train/*/*_L{}.tfrecords".format(1)
        )[170:200]

- Height of predownscaled image if NOT using multiple resolution records.
######
        dynamic_threshold_dict["image_predownscaled_height"] = 1024
        
- Width of predownscaled image if NOT using multiple resolution records.
######
        dynamic_threshold_dict["image_predownscaled_width"] = 1024

- Number of examples in one epoch of dynamic threshold training set.
    - "train_dataset_length" = (#slides * #image_patches per slide // batch_size * #gpu) * (batch_size * #gpu)
######
        dynamic_threshold_dict["train_dataset_length"] = 960

Training parameters.

- GCS location to write checkpoints, loss logs, and export models.
######
        training_dict["output_dir"] = "gs://my-bucket/trained_models/experiment"

Full parameters.

- Set final image size as a multiple of 2, starting at 4.
######
        image_size = 512

Overrides

        overrides = {
            "training": {
                "output_dir": "gs://{BCKT}/{ROOT}/{DT}_{DATA}_{HW}_{SZ}_{BS}_{FILES}_{LEN}_{EXPORT}_{MISC}".format(
                    BCKT=BUCKET,
                    ROOT=".../trained_models",
                    DATA="BACH",
                    SZ="512x512",
                )
            }
        }
        
        os.environ["IMAGE_SIZE"] = str(512)
        os.environ["JSON_CONFIG_GCS_PATH"] = "gs://.../model_json_config.json"

Bash Scripts

        %%bash
        JOBNAME=BACH_proganomaly_${IMAGE_SIZE}x${IMAGE_SIZE}_$(date -u +%y%m%d_%H%M%S)
        echo ${OUTPUT_DIR} ${REGION} ${JOBNAME}
        gcloud ai-platform jobs submit training ${JOBNAME} \
            --region=${REGION} \
            --module-name=trainer.task \
            --package-path=$PWD/../proganomaly_modules/training_module/trainer \
            --job-dir=gs://${BUCKET} \
            --staging-bucket=gs://${BUCKET} \
            --config=config.yaml \
            --runtime-version=${TFVERSION} \
            --python-version=${PYTHON_VERSION} \
            -- \
            --job-dir=./tmp \
            --json_config_gcs_path=${JSON_CONFIG_GCS_PATH} \
            --json_overrides=${JSON_OVERRIDES}