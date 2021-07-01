# Proganomaly Polygon Confusion Matrices Pipeline

## License

Copyright 2020 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at [Apache License Page](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

## Required Package Installation
- Required package installation commands
######
        sudo apt-get update
        pip3 install --upgrade pip
        pip3 install tensorflow
        pip3 install scikit-image
        pip3 install shapely

## Polygon Confusion Matrices Config Introduction

There are four main groups of configs: input, output, polygon, and Dataflow.

The input configs are where input files are located. Annotations and KDE grayscale images are converted to Shapely 
`MultiPolygon`s. The patch coordinates are used to create a `MultiPolygon` of all of the patches as a outer bound of 
the prediction `MultiPolygon`.

The output config contains a boolean flag whether to use Dataflow or not and the output GCS path to write the polygon 
confusion matrix results.

The polygon config contains all hyperparamters needed for making polygons.

The dataflow config contains the parameters needed for the Dataflow polygon pipeline, only used if Dataflow gets called.

## Polygon Confusion Matrices Parameters

input config

- name of the slide run the inference on
######
        "slide_name": "",
        
- gcs path of the annotated image
######
        "annotations_image_gcs_path": "",
        
- gcs path of the KDE grey scale image returned by inference pipeline
######
        "kde_gs_image_gcs_path": "",
    
output_config 

- whether or not to use GCS Dataflow to perform the polygon confusion matrices pipeline
    - Only specify output_gcs_path if running a beam stitch job. Don't forget the slash on the end.
######
        "use_dataflow": True,
        "output_gcs_path": "gs://path to polygon confusion matrice output file(s)/"

polygon_config 

- Height in pixels of a patch.
######
        "patch_height": 1024,

- Width in pixels of a patch.
######
        "patch_width": 1024,

- Height in pixels of input image.
######
        "image_height": 8192,

- Width in pixels of input image.
######
        "image_width": 8192,

- Let's say you have a slide that is 86000 x 112000. This means, if my patches are 1024 x1024, that 83.984 ~ 83 
patches can fit in the x dimension and 109.375 ~ 109 patches can fit in the y dimension. However, I need to stitch 
cleanly a left and a right patch (power of 2 in the x dimension) and an up and a down patch (power of 2 in the y 
dimension). Therefore the next closest biggest power of 2 in the x dimension is 83 -> 128 and in the y is 109 -> 128. 
This results in a 128 x 128 patch image. Even though this is already square, in case it is not, we take the max of each 
dimension and then set both to that.

- log(128, 2) = 7. That is where the 7 comes from. The stitching will require a depth of 7 of the 4-ary tree to 
complete the slide.
    - Depth: 7, Size 128x128
    - Depth: 6, Size 64x64
    - Depth: 5, Size 32x32
    - Depth: 4, Size 16x16
    - Depth: 3, Size 8x8
    - Depth: 2, Size 4x4
    - Depth: 1, Size 2x2
    - Depth: 0, Size 1x1
    
I need to be able to reconstruct that size. Therefore, (num_patches) * (patch_size) = (2 ** 7) * (1024)
######
        "effective_slide_height": 2 ** 7 * 1024,
        "effective_slide_width": 2 ** 7 * 1024,
        
- Max number of seconds to wait for each element to complete. This is helps Dataflow jobs from getting stuck and 
failing due to straggelers. To disable, set to 0.
######
        "timeout": 600,
        
- List of thresholds to apply to KDE grayscale image to use result to create Polygons.
######
        "thresholds": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 
        0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25],
        
- List of dilation factors to scale Polygons.
######
        "dilation_factors": [1.0, 1.1, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 25.0, 30.0]

dataflow_config

- Project to run the Dataflow job.
######
        "project": "",

- GCS bucket to stage temporary files.
######
        "bucket": "",

- Region to run the Dataflow job, make sure you have quota.
######
        "region": "",

- Autoscaling mode for Dataflow job. Possible values are THROUGHPUT_BASED to enable autoscaling or NONE to disable.
######
        "autoscaling_algorithm": "NONE",

- Initial number of Google Compute Engine instances to use when executing your pipeline. This option determines how 
many workers the Dataflow service starts up when your job begins.
######
        "num_workers": 60,

- Compute Engine machine type that Dataflow uses when starting worker VMs.
######
        "machine_type": "",

- Disk size, in gigabytes, to use on each remote Compute Engine worker instance.
######
        "disk_size_gb": 1000,

- Specifies a user-managed controller service account, using the format my-service-account-name@<project-id>.iam.
gserviceaccount.com.
######
        "service_account_email": "",

- Specifies whether Dataflow workers use public IP addresses. If the value is set to false, Dataflow workers use 
private IP addresses for all communication. In this case, if the subnetwork option is specified, the network option is 
ignored. Make sure that the specified network or subnetwork has Private Google Access enabled. Public IP addresses have 
an associated cost.
######
        "use_public_ips": False,

- Compute Engine network for launching Compute Engine instances to run your pipeline.
######
        "network": "",

- Compute Engine subnetwork for launching Compute Engine instances to run your pipeline.
######
        "subnetwork": "",

- Runner of pipeline. "DirectRunner" for running local, "DataflowRunner" for running distributed Dataflow job.
######
        # Directrunner or DataflowRunner
        "runner": "DataflowRunner"  

config 
- entire inference pipeline config
######
        "input": input_config,
        "output": output_config,
        "polygon": polygon_config,
        "dataflow": dataflow_config
