# Proganomaly Inference Pipeline

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
        %%bash
        sudo apt-get update
        sudo apt-get --assume-yes install openslide-tools
        sudo apt-get --assume-yes install python-openslide
        pip3 install --upgrade pip
        pip3 install tensorflow
        pip3 install opencv-python-headless
        pip3 install openslide-python
        pip3 install matplotlib
        pip3 install scikit-image
        pip3 install shapely

## Inference Config Introduction

There are six main groups of configs: input, output, GAN inference, segmentation inference, polygon inference, and 
Dataflow.

The input configs are where input files are located. As requested there are several possible paths.

1. If `pre_computed_image_gcs_path` is not empty, inference from the model will be skipped and images at that GCS 
path will be read and used to create polygons.
1. If `png_individual_gcs_glob_pattern` is not empty, query image patch files will be read from GCS using this glob 
pattern and will call the model for inference and will create polygons at the end. To increase speed of inference, we 
can copy the SavedModel to a local directory and set `export_dir` to that path. Don't forget to then change 
`exports_on_gcs` to `True`.
1. If `png_patch_stitch_gcs_path` or `wsi_stitch_gcs_path` is not empty then the Dataflow stitching pipeline will be 
called where all of the patches will call the model for inference and then be stitched together. Since this can take 
quite some time, the notebook may lose connection, despite the Dataflow job still running perfectly in the background. 
If this happens, then change `pre_computed_image_gcs_path` to be what `gcs_output_image_filepattern` was previously to 
use the results of the Dataflow pipeline. `png_patch_stitch_gcs_path` and `wsi_stitch_gcs_path` cannot both be set at 
the same time.
    1. If `png_patch_stitch_gcs_path` is not empty, then the Dataflow stitching pipeline will use pre-extracted patch 
    PNG files at the GCS path for inference and then stitch after.
    1. If `wsi_stitch_gcs_path` is not empty, then the Dataflow stitching pipeline will use the WSI at that path to 
    extract patches in memory and then stitch after inference.

The output config contains boolean flags for which output types are desired as well 
as the GCS output path `gcs_output_image_filepattern`.

The GAN inference config contains all hyperparamters needed for GAN inference.

The segmentation inference config contains all hyperparamters needed for segmentation inference.

The polygon inference config contains all hyperparamters needed for polygon inference.

The dataflow config contains the parameters needed for the Dataflow sitching pipeline, only used if Dataflow gets 
called.

## Inference Parameters

input config

- name of the slide run the inference on
######
        "slide_name": "",
        
- whether or not use pre-computed inference output files without calling pre-trained GAN model to do the inference
######
        "use_pre_computed_gcs_paths": True,
        
- pre-computed inference output files path, leave the blank if you do not want output certain files
######
        "pre_computed_query_images_gcs_path": "",
        "pre_computed_query_gen_encoded_images_gcs_path": "",
        "pre_computed_query_anomaly_images_linear_rgb_gcs_path": "",
        "pre_computed_query_anomaly_images_linear_gs_gcs_path": "",
        "pre_computed_query_mahalanobis_distance_images_linear_gcs_path": "",
        "pre_computed_query_pixel_anomaly_flag_images_gcs_path": "",
        "pre_computed_kde_rgb_gcs_path": "",
        "pre_computed_kde_gs_gcs_path": "",
        "pre_computed_kde_gs_thresholded_gcs_path": "",
        "pre_computed_segmentation_cell_coords_gcs_path": "",
        "pre_computed_segmentation_nuclei_coords_gcs_path": "",
        "pre_computed_patch_coordinates_gcs_path": "",
        
- gcs path for individual or multiple image patches need to run inference on, make sure you have enough memory to 
provide multiple image patches
    - make sure your individual (/multiple) image patch file names follow the format as {slide_name}.{slide_format}_x_{x_coordinate_value}_y_{y coordinate value}_width_{image_patch_width}_height_{image_patch_height}.{image_patch_file_format}
######    
        "png_individual_gcs_glob_pattern": "",
- gcs path for image patches from certain slide need to run inference on through Dataflow and perform stitching 
afterwards
######
        "png_patch_stitch_gcs_glob_pattern": "",
- gcs path for entire slide needs to run inference on through Dataflow and perform stitching 
afterwards
######
         "wsi_stitch_gcs_path": ""

output_config 

- whether or not to output certain output files after the completion of the inference pipeline 
    - Require query_images, kde_gs, and patch_coordinates to be True if needs the output of kde_gs_polygon
    - Only specify output_gcs_path if running a beam stitch job. Don't forget the slash on the end.
######
        "output_query_images": True,
        "output_query_gen_encoded_images": True,
        "output_query_anomaly_images_linear_rgb": True,
        "output_query_anomaly_images_linear_gs": True,
        "output_query_mahalanobis_distance_images_linear": True,
        "output_query_pixel_anomaly_flag_images": True,
        "output_kde_rgb": True,
        "output_kde_gs": True,
        "output_kde_gs_thresholded": True,
    
        "output_kde_gs_polygon": True,
        "output_segmentation_cell_coords": False,
        "output_segmentation_nuclei_coords": False,
        "output_patch_coordinates": False,
      
        "output_gcs_path": "gs://path to inference output files/"

gan_inference_config 

- Approximate width of resultant thumbnail image.
######
        "target_image_width": 500,
        
- Method to use for converting thumbnail of slide into binary mask. Either otsu or rgb2hed.
######
    "thumbn ail_method": "otsu",
    
- Threshold to convert RGB2HED image to binary mask.
######
        "rgb2hed_threshold": -0.41,
    
- Threshold to compare with percent of binary flags within a patch region to include in collection.
######    
        "include_patch_threshold": 0.5,
    
- Whether SavedModel exports are on GCS or local.
######
        "exports_on_gcs": True,
    
- GAN SavedModel export directory on GCS or local filesystem.
######
        "gan_export_dir": "",
        
- Folder name of exported GAN SavedModel.
######
        "gan_export_name": "dynamic_threshold",
    
- Bandwidth of the kernel.
######
        "bandwidth": 100.0,

- Kernel to use for density estimation.
######
        "kernel": "gaussian",

- Distance metric to use. Note that not all metrics are valid with all algorithms.
######
        "metric": "euclidean",

- Number of sample bins to create in the x dimension.
######
        "xbins": 100,

- Number of sample bins to create in the y dimension.
######
        "ybins": 100,

- Minimum number of adjacent points as not to be removed from image.
######
        "min_neighborhood_count": 50,

- Connectivity defining the neighborhood of a pixel.
######
        "connectivity": 10,

- Minimum number of anomaly points following removing small objects to not clear all flags.
######
        "min_anomaly_points_remaining": 200,

- Exponent to use for scaling.
######
        "scaling_power": 0.5,

- Positive factor to scale anomaly flag counts by.
######
        "scaling_factor": 100000.0,

- Which color map to use.
######
        "cmap_str": "turbo",

- Threshold to convert KDE grayscale image into binary mask to create image.
######
        "kde_threshold": 0.2,

- Threshold to override learned Mahalanobis distance threshold from SavedModel for creating Mahalanobis binary mask.
######
        "custom_mahalanobis_distance_threshold": 2.8172882 + 3.0 * 67.66342,  # mu + num_sigma * stddev
    
- Depth of n-ary tree for GAN image stitching. Let's say you have a slide that is 86000 x 112000. This means, if my 
patches are 1024 x1024, that 83.984 ~ 83 patches can fit in the x dimension and 109.375 ~ 109 patches can fit in 
the y dimension. However, I need to stitch cleanly a left and a right patch (power of 2 in the x dimension) and an up 
and a down patch (power of 2 in the y dimension). Therefore the next closest biggest power of 2 in the x dimension is 
83 -> 128 and in the y is 109 -> 128. This results in a 128 x 128 patch image. Even though this is already square, in 
case it is not, we take the max of each dimension and then set both to that.

- That is where the 7 comes from. The stitching will log(128, 2) = 7 require a depth of 7 of the 4-ary tree to complete 
the slide.
    - Depth: 7, Size 128x128
    - Depth: 6, Size 64x64
    - Depth: 5, Size 32x32
    - Depth: 4, Size 16x16
    - Depth: 3, Size 8x8
    - Depth: 2, Size 4x4
    - Depth: 1, Size 2x2
    - Depth: 0, Size 1x1
######
        # most use 7
        "nary_tree_depth": 7  

segmentation_inference_config

Segmentation inference config only used if outputting:
######
        output_segmentation_cell_coords 
######      
or 
######
        output_segmentation_nuclei_coords.
 
- Directory containing exported segmentation models.
######
        "segmentation_export_dir": "",

- Name of segmentation model
######
        "segmentation_model_name": "",

- Size of each patch of image for segmentation model.
######
        "segmentation_patch_size": 128,

- Number of pixels to skip for each patch of image for segmentation model.
######
        "segmentation_stride": 16,

- Whether to median blur images before segmentation.
######
        "segmentation_median_blur_image": False,

- Kernel size of median blur for segmentation.
######
        "segmentation_median_blur_kernel_size": 9,

- Number of patches to include in a group for segmentation.
######
        "segmentation_group_size": 10

inference_config 

- Number of images to include in each batch for inference.
######
        "batch_size": 8,

- Height in pixels of a patch.
######
        "patch_height": 1024,

- Width in pixels of a patch.
######
        "patch_width": 1024,

- Number of channels of an image patch.
######
        "patch_depth": 3,

polygon_config 

- Threshold to convert KDE grayscale image into binary mask to create Polygon.
######
        "kde_gs_polygon_threshold": 0.2,
        
- Factor to scale/dilate the `MultiPolygon`.
###### 
        "kde_gs_polygon_dilation_factor": 1.0,
        
- The origin each polygon should be scaled about. 'center' or 'centroid'.
######
        "dilation_origin": "centroid",
        
- Whether to limit Polygon vertices to only lie within patches. Requires patch coordinates list. For stitched slide 
images, set to True. Otherwise, for individual PNG images, set to False.
######
        "limit_polygon_vertices_to_only_patches": True,
        
- Let's say you have a slide that is 86000 x 112000. This means, if my patches are 1024 x1024, that 83.984 ~ 83 patches 
can fit in the x dimension and 109.375 ~ 109 patches can fit in the y dimension. However, I need to stitch cleanly a 
left and a right patch (power of 2 in the x dimension) and an up and a down patch (power of 2 in the y dimension). 
Therefore the next closest biggest power of 2 in the x dimension is 83 -> 128 and in the y is 109 -> 128. This results 
in a 128 x 128 patch image. Even though this is already square, in case it is not, we take the max of each dimension 
and then set both to that.

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
        "effective_slide_width": 2 ** 7 * 1024

dataflow_config

Dataflow config only used if calling a Dataflow stitch job.

- Project to run the Dataflow job .
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

- Initial number of Google Compute Engine instances to use when executing your pipeline. This option determines 
how many workers the Dataflow service starts up when your job begins.
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
private IP addresses for all communication. In this case, if the subnetwork option is specified, the network option 
is ignored. Make sure that the specified network or subnetwork has Private Google Access enabled. Public IP addresses 
have an associated cost.
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
        "inference": inference_config,
        "polygon": polygon_config,
        "dataflow": dataflow_config
