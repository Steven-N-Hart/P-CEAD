# Copyright 2021 Google Inc. All Rights Reserved.
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

import apache_beam as beam
from apache_beam.io.gcp import gcsio
from collections import defaultdict
import cv2
import math
import matplotlib.pyplot as plt
import openslide
import os
import numpy as np
import tensorflow as tf


class InferenceDoFn(beam.DoFn):
    """ParDo class that performs inference on image patch.

    Attributes:
        wsi_stitch_gcs_path: str, GCS path of WSI file.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        patch_depth: int, the number of channels of an image patch.
        gan_export_dir: str, directory where the exported trained SavedModel
            resides.
        gan_export_name: str, name of exported trained SavedModel folder.
        generator_architecture: str, name of generator architecture, either
            'berg' or 'GANomaly'.
        berg_use_Z_inputs: bool, for berg architecture, whether to use Z
            inputs. Query image inputs are always used.
        berg_latent_size: int, for berg architecture, the latent size of the
            noise vector.
        berg_latent_mean: float, for berg architecture, the latent vector's
            random normal mean.
        berg_latent_stddev: float, for berg architecture, the latent vector's
            random normal standard deviation.
        image_stitch_types_set: set, strings of which image types to stitch.
        bandwidth: float, the bandwidth of the kernel.
        kernel: str, the kernel to use for density estimation.
        metric: str, the distance metric to use. Note that not all metrics
            are valid with all algorithms.
        xbins: int, number of sample bins to create in the x dimension.
        ybins: int, number of sample bins to create in the y dimension.
        min_neighborhood_count: int, minimum number of adjacent points as
            not to be removed from image.
        connectivity: int, connectivity defining the neighborhood of a pixel.
        min_anomaly_points_remaining: int, minimum number of anomaly points
            following removing small objects to not clear all flags.
        scaling_power: float, the exponent to use for scaling.
        scaling_factor: float, positive factor to scale anomaly flag
            counts by.
        cmap_str: str, which color map to use.
        dynamic_bandwidth_scale_factor: float, amount to scale the
            bandwidth based on anomaly counts.
        max_anomaly_points_for_kde: int, maximum number of points allowed
            to run KDE. Otherwise entire image is marked as anomalous.
        kde_threshold: float, threshold to convert KDE grayscale image into
            binary mask.
        annotation_patch_gcs_filepath: str, GCS path where annotation patch
            images are stored.
        num_confusion_matrix_thresholds: int, number of thresholds to
            calculate confusion matrix metrics over for comparing binary KDE
            masks with annotations.
        custom_mahalanobis_distance_threshold: float, threshold to override
            learned Mahalanobis distance threshold from SavedModel for
            creating Mahalanobis binary mask.
        segmentation_coord_types_set: set, strings of which segmentation types
            to output.
        segmentation_export_dir: str, directory containing exported
            segmentation models.
        segmentation_model_name: str, name of segmentation model.
        segmentation_patch_size: int, size of each patch of image for
            segmentation model.
        segmentation_stride: int, number of pixels to skip for each patch of
            image for segmentation model.
        segmentation_median_blur_image: bool, whether to median blur images
            before segmentation.
        segmentation_median_blur_kernel_size: int, kernel size of median blur
            for segmentation.
        segmentation_group_size: int, number of patches to include in a group
            for segmentation.
    """
    def __init__(
        self,
        wsi_stitch_gcs_path,
        patch_height,
        patch_width,
        patch_depth,
        gan_export_dir,
        gan_export_name,
        generator_architecture,
        berg_use_Z_inputs,
        berg_latent_size,
        berg_latent_mean,
        berg_latent_stddev,
        image_stitch_types_set,
        bandwidth,
        kernel,
        metric,
        xbins,
        ybins,
        min_neighborhood_count,
        connectivity,
        min_anomaly_points_remaining,
        scaling_power,
        scaling_factor,
        cmap_str,
        dynamic_bandwidth_scale_factor,
        max_anomaly_points_for_kde,
        kde_threshold,
        annotation_patch_gcs_filepath,
        num_confusion_matrix_thresholds,
        custom_mahalanobis_distance_threshold,
        segmentation_coord_types_set,
        segmentation_export_dir,
        segmentation_model_name,
        segmentation_patch_size,
        segmentation_stride,
        segmentation_median_blur_image,
        segmentation_median_blur_kernel_size,
        segmentation_group_size
    ):
        """Constructor of ParDo class that performs inference on image patch.

        Args:
            wsi_stitch_gcs_path: str, GCS path of WSI file.
            patch_height: int, the height in pixels of an image patch.
            patch_width: int, the width in pixels of an image patch.
            patch_depth: int, the number of channels of an image patch.
            gan_export_dir: str, directory where the exported trained
                SavedModel resides.
            gan_export_name: str, name of exported trained SavedModel folder.
            generator_architecture: str, name of generator architecture,
                either 'berg' or 'GANomaly'.
            berg_use_Z_inputs: bool, for berg architecture, whether to use Z
                inputs. Query image inputs are always used.
            berg_latent_size: int, for berg architecture, the latent size of
                the noise vector.
            berg_latent_mean: float, for berg architecture, the latent
                vector's random normal mean.
            berg_latent_stddev: float, for berg architecture, the latent
                vector's random normal standard deviation.
            image_stitch_types_set: set, strings of which image types to
                stitch.
            bandwidth: float, the bandwidth of the kernel.
            kernel: str, the kernel to use for density estimation.
            metric: str, the distance metric to use. Note that not all metrics
                are valid with all algorithms.
            xbins: int, number of sample bins to create in the x dimension.
            ybins: int, number of sample bins to create in the y dimension.
            min_neighborhood_count: int, minimum number of adjacent points as
                not to be removed from image.
            connectivity: int, connectivity defining the neighborhood of a
                pixel.
            min_anomaly_points_remaining: int, minimum number of anomaly
                points following removing small objects to not clear all
                flags.
            scaling_power: float, the exponent to use for scaling.
            scaling_factor: float, positive factor to scale anomaly flag
                counts by.
            cmap_str: str, which color map to use.
            dynamic_bandwidth_scale_factor: float, amount to scale the
                bandwidth based on anomaly counts.
            max_anomaly_points_for_kde: int, maximum number of points allowed
                to run KDE. Otherwise entire image is marked as anomalous.
            kde_threshold: float, threshold to convert KDE grayscale image
                into binary mask.
            annotation_patch_gcs_filepath: str, GCS path where annotation
                patch images are stored.
            num_confusion_matrix_thresholds: int, number of thresholds to
                calculate confusion matrix metrics over for comparing binary
                KDE masks with annotations.
            custom_mahalanobis_distance_threshold: float, threshold to
                override learned Mahalanobis distance threshold from
                SavedModel for creating Mahalanobis binary mask.
            segmentation_coord_types_set: set, strings of which segmentation
                types to output.
            segmentation_export_dir: str, directory containing exported
                segmentation models.
            segmentation_model_name: str, name of segmentation model.
            segmentation_patch_size: int, size of each patch of image for
                segmentation model.
            segmentation_stride: int, number of pixels to skip for each patch
                of image for segmentation model.
            segmentation_median_blur_image: bool, whether to median blur
                images before segmentation.
            segmentation_median_blur_kernel_size: int, kernel size of median
                blur for segmentation.
            segmentation_group_size: int, number of patches to include in a
                group for segmentation.
        """
        self.wsi_stitch_gcs_path = wsi_stitch_gcs_path
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_depth = patch_depth
        self.gan_export_dir = gan_export_dir
        self.gan_export_name = gan_export_name
        self.generator_architecture = generator_architecture
        self.berg_use_Z_inputs = berg_use_Z_inputs
        self.berg_latent_size = berg_latent_size
        self.berg_latent_mean = berg_latent_mean
        self.berg_latent_stddev = berg_latent_stddev
        self.image_stitch_types_set = image_stitch_types_set
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.xbins = xbins
        self.ybins = ybins
        self.min_neighborhood_count = min_neighborhood_count
        self.connectivity = connectivity
        self.min_anomaly_points_remaining = (
            min_anomaly_points_remaining
        )
        self.scaling_power = scaling_power
        self.scaling_factor = scaling_factor
        self.cmap_str = cmap_str
        self.dynamic_bandwidth_scale_factor = dynamic_bandwidth_scale_factor
        self.max_anomaly_points_for_kde = max_anomaly_points_for_kde
        self.kde_threshold = kde_threshold
        self.annotation_patch_gcs_filepath = annotation_patch_gcs_filepath
        self.num_confusion_matrix_thresholds = num_confusion_matrix_thresholds
        self.custom_mahalanobis_distance_threshold = (
            custom_mahalanobis_distance_threshold
        )
        self.segmentation_coord_types_set = segmentation_coord_types_set
        self.segmentation_export_dir = segmentation_export_dir
        self.segmentation_model_name = segmentation_model_name
        self.segmentation_patch_size = segmentation_patch_size
        self.segmentation_stride = segmentation_stride
        self.segmentation_median_blur_image = segmentation_median_blur_image
        self.segmentation_median_blur_kernel_size = (
            segmentation_median_blur_kernel_size
        )
        self.segmentation_group_size = segmentation_group_size

    def process_non_patch_grid_elements(self, grid_dict_list, gs_image_set):
        """Processes non-patch grid elements.

        Args:
            grid_dict_list: list, contains dicts of coordinates and 4-ary tree
                grid indices.
            gs_image_set: set, image output types that are grayscale, i.e.
                have only one channel.

        Returns:
            List of dictionaries of image arrays.
        """
        image_dict_list = []
        for grid_dict in grid_dict_list:
            image_dict = {}
            for stitch_type in self.image_stitch_types_set:
                if stitch_type in gs_image_set:
                    image_dict[stitch_type] = tf.ones(
                        shape=(self.patch_height, self.patch_width, 1),
                        dtype=tf.float32
                    )
                else:
                    image_dict[stitch_type] = tf.ones(
                        shape=(
                            self.patch_height,
                            self.patch_width,
                            self.patch_depth
                        ),
                        dtype=tf.float32
                    )
            image_dict_list.append(image_dict)
        return image_dict_list

    def get_query_images(self, grid_dict_list):
        """Processes patch grid elements.

        Args:
            grid_dict_list: list, contains dicts of coordinates and 4-ary tree
                grid indices.

        Returns:
            List containing query image tensors.
        """
        images = []
        if self.wsi_stitch_gcs_path:
            gcs = gcsio.GcsIO()
            local_file = "slide_file.svs"
            num_retries = 0
            while num_retries < 100:
                try:
                    with open(local_file, "wb") as f:
                        f.write(gcs.open(self.wsi_stitch_gcs_path).read())
                    if tf.io.gfile.exists(path=local_file):
                        wsi = openslide.OpenSlide(filename=local_file)
                        images = []
                        for grid_dict in grid_dict_list:
                            image = np.array(
                                wsi.read_region(
                                    location=grid_dict["coords"],
                                    level=0,
                                    size=(self.patch_width, self.patch_height)
                                )
                            )[:, :, :3]
                            images.append(image)
                except:
                    num_retries += 1
                else:
                    break
        else:
            for grid_dict in grid_dict_list:
                raw_image = tf.io.read_file(filename=grid_dict["filename"])
                image = tf.io.decode_png(
                    contents=raw_image, channels=self.patch_depth
                )
                image = tf.image.rot90(image=image, k=2)
                images.append(image)
        return images

    ##########################################################################
    ### GAN ##################################################################
    ##########################################################################
        
    def scale_images(self, images):
        """Scales images from [0, 255] to [-1., 1.].

        Args:
            images: np.array, array of images with range [0, 255] of shape
                (num_images, height, width, num_channels).

        Returns:
            Tensor of images with range [-1., 1.] of shape
                (num_images, height, width, num_channels).
        """
        return tf.cast(x=images, dtype=tf.float32) * (2. / 255) - 1.

    def get_saved_model_serving_signatures(self, export_name, params):
        """Gets SavedModel's serving signatures for inference.

        Args:
            export_name: str, name of exported SavedModel.
            params: dict, user passed parameters.

        Returns:
            Loaded SavedModel and its serving signatures for inference.
        """
        print(
            "get_saved_model_serving_signatures: output_dir = {}, export_name = {}".format(
                params["output_dir"], export_name
            )
        )
        loaded_model = tf.saved_model.load(
            export_dir=os.path.join(
                params["output_dir"], "export", export_name
            )
        )

        infer = loaded_model.signatures["serving_default"]

        # Loaded model also needs to be returned so that infer can find the
        # variables within the graph in the outer scope.
        return loaded_model, infer

    def create_export_bool_lists(self, params):
        """Creates lists of user parameters bools for exporting.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of bools relating to the Z serving input and list of bools
                relating to the query images serving input.
        """
        export_Z_bool_list = [
            params["export_Z"],
            params["export_generated_images"],
            params["export_encoded_generated_logits"],
            params["export_encoded_generated_images"]
        ]

        export_query_image_bool_list = [
            params["export_query_images"],
            params["export_query_encoded_logits"],
            params["export_query_encoded_images"],
            params["export_query_gen_encoded_logits"],
            params["export_query_gen_encoded_images"],
            params["export_query_enc_encoded_logits"],
            params["export_query_enc_encoded_images"],
            params["export_query_anomaly_images_sigmoid"],
            params["export_query_anomaly_images_linear"],
            params["export_query_mahalanobis_distances"],
            params["export_query_mahalanobis_distance_images_sigmoid"],
            params["export_query_mahalanobis_distance_images_linear"],
            params["export_query_pixel_anomaly_flag_images"],
            params["export_query_anomaly_scores"],
            params["export_query_anomaly_flags"]
        ]

        return export_Z_bool_list, export_query_image_bool_list

    def parse_predictions_dict(self, predictions, num_growths):
        """Parses predictions dictionary to remove graph generated suffixes.

        Args:
            predictions: dict, predictions dictionary directly from SavedModel
                inference call.
            num_growths: int, number of model growths contained in export.

        Returns:
            List of num_growths length of dictionaries with fixed keys and
                predictions.
        """
        predictions_by_growth = [{} for _ in range(num_growths)]

        for k in sorted(predictions.keys()):
            key_split = k.split("_")
            if key_split[-1].isnumeric() and key_split[-2].isnumeric():
                idx = 0 if num_growths == 1 else int(key_split[-2])
                predictions_by_growth[idx].update(
                    {"_".join(key_split[3:-2]): predictions[k]}
                )
            else:
                idx = 0 if num_growths == 1 else int(key_split[-1])
                predictions_by_growth[idx].update(
                    {"_".join(key_split[3:-1]): predictions[k]}
                )

        del predictions

        return predictions_by_growth

    def get_current_growth_predictions(self, export_name, Z, query_images, params):
        """Gets predictions from exported SavedModel for current growth.

        Args:
            export_name: str, name of exported SavedModel.
            Z: tensor, random latent vector of shape
                (batch_size, generator_latent_size).
            query_images: tensor, real images to query the model with of shape
                (batch_size, height, width, num_channels).
            params: dict, user passed parameters.

        Returns:
            List of num_growths length of dictionaries with fixed keys and
                predictions.
        """
        loaded_model, infer = self.get_saved_model_serving_signatures(
            export_name, params
        )

        (export_Z_bool_list,
         export_query_image_bool_list) = self.create_export_bool_lists(params)

        if query_images is not None:
            image_size = query_images.shape[1]
            assert(image_size % 2 == 0)

        if params["generator_architecture"] == "berg":
            if Z is not None and any(export_Z_bool_list):
                if query_images is not None and any(export_query_image_bool_list):
                    kwargs = {
                        "generator_decoder_inputs": Z,
                        "encoder_{0}x{0}_inputs".format(image_size): (
                            query_images
                        )
                    }

                    predictions = infer(**kwargs)
                else:
                    predictions = infer(generator_decoder_inputs=Z)
            else:
                if query_images is not None and any(export_query_image_bool_list):
                    kwargs = {
                        "encoder_{0}x{0}_inputs".format(image_size): (
                            query_images
                        )
                    }

                    predictions = infer(**kwargs)
                else:
                    print("Nothing was exported, so nothing to infer.")
        elif params["generator_architecture"] == "GANomaly":
            if query_images is not None and any(export_query_image_bool_list):
                kwargs = {
                    "generator_encoder_{0}x{0}_inputs".format(image_size): (
                        query_images
                    )
                }

                predictions = infer(**kwargs)

        predictions_by_growth = self.parse_predictions_dict(
            predictions=predictions, num_growths=1
        )

        return predictions_by_growth

    def plot_all_exports(
        self, 
        Z,
        query_images,
        exports_on_gcs,
        export_start_idx,
        export_end_idx,
        max_size,
        only_output_growth_set,
        num_rows,
        params
    ):
        """Plots predictions based on bool conditions.

        Args:
            Z: tensor, random latent vector of shape
                (batch_size, generator_latent_size).
            query_images: tensor, real images to query the model with of shape
                (batch_size, height, width, num_channels).
            exports_on_gcs: bool, whether exports are stored on GCS or locally.
            export_start_idx: int, index to start at in export list.
            export_end_idx: int, index to end at in export list.
            max_size: int, the maximum image size within the exported SavedModel.
            only_output_growth_set: set, which growth blocks to output.
            num_rows: int, number of rows to plot for each desired output.
            params: dict, user passed parameters.
        """
        predictions_by_growth = self.get_current_growth_predictions(
            export_name=self.gan_export_name,
            Z=Z,
            query_images=self.scale_images(query_images),
            params=params
        )

        return predictions_by_growth

    def plot_all_exports_by_architecture(
        self, 
        Z,
        query_images,
        exports_on_gcs,
        export_start_idx,
        export_end_idx,
        max_size,
        only_output_growth_set,
        num_rows,
        generator_architecture,
        overrides
    ):
        """Plots predictions based on bool conditions and architecture.

        Args:
            Z: tensor, random latent vector of shape
                (batch_size, generator_latent_size).
            query_images: tensor, real images to query the model with of shape
                (batch_size, height, width, num_channels)
            exports_on_gcs: bool, whether exports are stored on GCS or locally.
            export_start_idx: int, index to start at in export list.
            export_end_idx: int, index to end at in export list.
            max_size: int, the maximum image size within the exported SavedModel.
            only_output_growth_set: set, which growth blocks to output.
            num_rows: int, number of rows to plot for each desired output.
            generator_architecture: str, architecture to be used for generator,
                berg or GANomaly.
            overrides: dict, user passed parameters to override default config.
        """
        shared_config = {
            "generator_architecture": generator_architecture,
            "output_dir": "trained_models",
            "export_all_growth_phases": False,

            "export_query_images": True,

            "export_query_anomaly_images_sigmoid": True,
            "export_query_anomaly_images_linear": True,

            "export_query_mahalanobis_distances": False,
            "export_query_mahalanobis_distance_images_sigmoid": False,
            "export_query_mahalanobis_distance_images_linear": False,

            "export_query_pixel_anomaly_flag_images": False,

            "export_query_anomaly_scores": True,
            "export_query_anomaly_flags": True,

            "output_Z": False,
            "output_generated_images": False,
            "output_encoded_generated_logits": False,
            "output_encoded_generated_images": False,

            "output_query_images": False,

            "output_query_encoded_logits": False,
            "output_query_encoded_images": False,

            "output_query_gen_encoded_logits": False,
            "output_query_gen_encoded_images": False,
            "output_query_enc_encoded_logits": False,
            "output_query_enc_encoded_images": False,

            "output_query_anomaly_images_sigmoid": False,
            "output_query_anomaly_images_linear": False,

            "output_query_mahalanobis_distances": False,
            "output_query_mahalanobis_distance_images_sigmoid": False,
            "output_query_mahalanobis_distance_images_linear": False,

            "output_query_pixel_anomaly_flag_images": False,

            "output_query_anomaly_scores": False,
            "output_query_anomaly_flags": False,

            "output_transition_growths": False,
            "output_stable_growths": True,
            "image_depth": 3
        }

        if generator_architecture == "berg":
            params={
                "export_Z": True,
                "export_generated_images": True,
                "export_encoded_generated_logits": True,
                "export_encoded_generated_images": True,

                "export_query_encoded_logits": True,
                "export_query_encoded_images": True,

                "export_query_gen_encoded_logits": False,
                "export_query_gen_encoded_images": False,
                "export_query_enc_encoded_logits": False,
                "export_query_enc_encoded_images": False
            }
        elif generator_architecture == "GANomaly":
            Z = None

            params={
                "export_Z": False,
                "export_generated_images": False,
                "export_encoded_generated_logits": False,
                "export_encoded_generated_images": False,

                "export_query_encoded_logits": False,
                "export_query_encoded_images": False,

                "export_query_gen_encoded_logits": True,
                "export_query_gen_encoded_images": True,
                "export_query_enc_encoded_logits": True,
                "export_query_enc_encoded_images": True
            }

        params.update(shared_config)

        for key in overrides.keys():
            if key in params:
                params[key] = overrides[key]

        return self.plot_all_exports(
            Z=Z,
            query_images=query_images,
            exports_on_gcs=exports_on_gcs,
            export_start_idx=export_start_idx,
            export_end_idx=export_end_idx,
            max_size=max_size,
            only_output_growth_set=only_output_growth_set,
            num_rows=num_rows,
            params=params
        )

    def kde2D(
        self,
        x,
        y,
        bandwidth,
        kernel,
        metric,
        xbins,
        ybins,
        dynamic_bandwidth_scale_factor,
        **kwargs
    ):
        """Builds 2D kernel density estimate (KDE).

        Args:
            x: np.array, array of x-coordinates of points of shape
                (num_anomaly_flags,).
            y: np.array, array of y-coordinates of points of shape
                (num_anomaly_flags,).
            bandwidth: float, the bandwidth of the kernel.
            kernel: str, the kernel to use for density estimation.
            metric: str, the distance metric to use. Note that not all metrics
                are valid with all algorithms.
            xbins: int, number of sample bins to create in the x dimension.
            ybins: int, number of sample bins to create in the y dimension.
            dynamic_bandwidth_scale_factor: float, amount to scale the
                bandwidth based on anomaly counts.
            kwargs: dict, any other keyword args to pass to
                sklearn.neighbors.KernelDensity.

        Returns:
            np.array of the log-likelihood of samples of shape (xbins, ybins).
        """
        import sklearn.neighbors
        xbins = complex(0, xbins)
        ybins = complex(0, ybins)

        # Create grid of sample locations.
        xx, yy = np.mgrid[x.min():x.max():xbins, 
                          y.min():y.max():ybins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train  = np.vstack([y, x]).T

        anomaly_counts = x.shape[0]
        if anomaly_counts > 0:
            bandwidth *= max(
                1.0, dynamic_bandwidth_scale_factor / anomaly_counts
            )

        kde_skl = sklearn.neighbors.KernelDensity(
            bandwidth=bandwidth,
            kernel=kernel,
            metric=metric,
            **kwargs
        )
        kde_skl.fit(xy_train)

        # Score_samples() returns the log-likelihood of the samples.
        z = np.exp(kde_skl.score_samples(xy_sample))
        return np.reshape(z, xx.shape)

    def get_kernel_density_estimates(
        self,
        anomaly_flags,
        depth,
        bandwidth,
        kernel,
        metric,
        xbins,
        ybins,
        min_neighborhood_count,
        connectivity,
        min_anomaly_points_remaining,
        scaling_power,
        scaling_factor,
        cmap_str,
        dynamic_bandwidth_scale_factor,
        max_anomaly_points_for_kde
    ):
        """Gets kernel density estimates for both RGB and grayscale.

        Args:
            anomaly_flags: tensor, binary anomaly flag images of shape
                (batch_size, height, width).
            depth: int, the number of color channels of the images.
            bandwidth: float, the bandwidth of the kernel.
            kernel: str, the kernel to use for density estimation.
            metric: str, the distance metric to use. Note that not all metrics
                are valid with all algorithms.
            xbins: int, number of sample bins to create in the x dimension.
            ybins: int, number of sample bins to create in the y dimension.
            min_neighborhood_count: int, minimum number of adjacent points as
                not to be removed from image.
            connectivity: int, connectivity defining the neighborhood of a
                pixel.
            min_anomaly_points_remaining: int, minimum number of anomaly
                points following removing small objects to not clear all
                flags.
            scaling_power: float, the exponent to use for scaling.
            scaling_factor: float, positive factor to scale anomaly flag
                counts by.
            cmap_str: str, which color map to use.
            dynamic_bandwidth_scale_factor: float, amount to scale the
                bandwidth based on anomaly counts.
            max_anomaly_points_for_kde: int, maximum number of points allowed
                to run KDE. Otherwise entire image is marked as anomalous.

        Returns:
            mesh_rgb: np.array, RGB KDE image of shape
                (batch_size, height, width, 3).
            mesh_gs: np.array, grayscale KDE image of shape
                (batch_size, height, width, 1).
        """
        import skimage.morphology
        batch_size = anomaly_flags.shape[0]

        # shape = (batch_size, 1).
        anomaly_flag_counts = tf.expand_dims(
            input=tf.reduce_sum(
                input_tensor=tf.cast(x=anomaly_flags == 1., dtype=tf.float64),
                axis=(1, 2)
            ),
            axis=-1
        )

        anomaly_flags_removed = []
        for image in anomaly_flags:
            morphed_image = skimage.morphology.remove_small_objects(
                ar=image.numpy() == 1.,
                min_size=min_neighborhood_count,
                connectivity=connectivity
            )
            morphed_image[0, 0] = 1.
            morphed_image[0, -1] = 1.
            morphed_image[-1, 0] = 1.
            morphed_image[-1, -1] = 1.
            anomaly_flags_removed.append(morphed_image)
        anomaly_flags = tf.cast(
            x=tf.stack(values=anomaly_flags_removed, axis=0), dtype=tf.float32
        )

        # shape = (batch_size, height, width, depth).
        tiled_anomaly_flags = tf.tile(
            input=tf.expand_dims(input=anomaly_flags, axis=-1),
            multiples=(1, 1, 1, depth)
        )

        counts_remaining = []
        zzs = []
        for i in range(batch_size):
            # shape = (num_true, 2).
            anomaly_points = tf.where(
                condition=tf.equal(x=tiled_anomaly_flags[i, :, :, 0], y=1.)
            ).numpy()
            counts_remaining.append(anomaly_points.shape[0])

            if anomaly_points.shape[0] > min_anomaly_points_remaining:
                if anomaly_points.shape[0] <= max_anomaly_points_for_kde:
                    # each shape = (num_true,).
                    x, y = anomaly_points[:, 0], anomaly_points[:, 1]

                    # shape = (xbins, ybins).
                    zz = self.kde2D(
                        x=x,
                        y=y,
                        bandwidth=bandwidth,
                        kernel=kernel,
                        metric=metric,
                        xbins=xbins,
                        ybins=ybins,
                        dynamic_bandwidth_scale_factor=(
                            dynamic_bandwidth_scale_factor
                        )
                    )
                else:
                    zz = tf.ones(shape=(xbins, ybins), dtype=tf.float64)
            else:
                zz = tf.zeros(shape=(xbins, ybins), dtype=tf.float64)

            zzs.append(zz)

        # shape = (batch_size, xbins, ybins).
        zz = tf.stack(values=zzs, axis=0)

        # shape = (batch_size, xbins * ybins).
        zz_flatten = tf.reshape(tensor=zz, shape=(batch_size, xbins * ybins))

        zz_flatten_scaled = zz_flatten

        # shape = (batch_size, xbins * ybins).
        zz_min_scaled = tf.math.reduce_min(
            input_tensor=zz_flatten_scaled, axis=-1, keepdims=True
        )

        # shape = (batch_size, xbins * ybins).
        zz_max_scaled = tf.math.reduce_max(
            input_tensor=zz_flatten_scaled, axis=-1, keepdims=True
        )

        # shape = (batch_size, xbins * ybins).
        zz_normalized_scaled = tf.math.divide_no_nan(
            x=zz_flatten_scaled - zz_min_scaled,
            y=zz_max_scaled - zz_min_scaled
        )

        # shape = (batch_size, xbins * ybins).
        zz_normalized_scaled = tf.minimum(
            x=zz_normalized_scaled * tf.pow(
                x=anomaly_flag_counts / scaling_factor, y=scaling_power
            ),
            y=1.
        )

        # shape = (batch_size, xbins * ybins).
        zz_int_flat = tf.cast(
            x=tf.math.round(x=zz_normalized_scaled * 255.),
            dtype=tf.int32
        )

        cmap_colors = plt.get_cmap(cmap_str).colors

        # shape = (256, 3).
        rgb_cm = tf.constant(value=cmap_colors, dtype=tf.float32)

        # shape = (batch_size, xbins * ybins, depth).
        zz_gathered = tf.gather(indices=zz_int_flat, params=rgb_cm)

        # shape = (batch_size, xbins, ybins, depth).
        zz_rgb = tf.reshape(
            tensor=zz_gathered, shape=(batch_size, xbins, ybins, depth)
        )

        # shape = (batch_size, xbins, ybins, depth).
        zz_rgb_scaled = zz_rgb * 2. - 1.

        # shape = (batch_size, height, width, depth).
        mesh_rgb = tf.image.resize(
            images=zz_rgb_scaled,
            size=(tiled_anomaly_flags.shape[1], tiled_anomaly_flags.shape[2])
        )

        # shape = (batch_size, xbins, ybins, 1).
        zz_gs = tf.reshape(
            tensor=zz_normalized_scaled, shape=(batch_size, xbins, ybins, 1)
        )

        # shape = (batch_size, xbins, ybins, depth).
        zz_gs_scaled = zz_gs * 2. - 1.

        # shape = (batch_size, height, width, 1).
        mesh_gs = tf.image.resize(
            images=zz_gs_scaled,
            size=(tiled_anomaly_flags.shape[1], tiled_anomaly_flags.shape[2])
        )

        return mesh_rgb, mesh_gs

    def confusion_matrix(self, filename, image, label_image):
        """Creates confusion matrix metrics at thresholds between images.

        Args:
            filename: str, filaname of image.
            image: np.array, image array of shape
                (patch_height, patch_width, 1).
            label_image: np.array, label image array of shape
                (patch_height, patch_width, 1).

        Returns:
            List of dictionaries containing confusion matrix metrics at
                thresholds.
        """
        thresholds = tf.linspace(
            start=0., stop=1., num=self.num_confusion_matrix_thresholds
        )
        images_thresholded = image > thresholds
        return [
            {
                "filename": filename,
                "threshold": th,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
            for th, tp, fp, fn, tn in zip(
                thresholds.numpy().tolist(),
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        x=tf.logical_and(
                            x=images_thresholded, y=label_image == 1
                        ),
                        dtype=tf.int32
                    ),
                    axis=(0, 1)
                ).numpy().tolist(),
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        x=tf.logical_and(
                            x=images_thresholded, y=label_image == 0
                        ),
                        dtype=tf.int32
                    ),
                    axis=(0, 1)
                ).numpy().tolist(),
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        x=tf.logical_and(
                            x=~images_thresholded, y=label_image == 1
                        ),
                        dtype=tf.int32
                    ),
                    axis=(0, 1)
                ).numpy().tolist(),
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        x=tf.logical_and(
                            x=~images_thresholded, y=label_image == 0
                        ),
                        dtype=tf.int32
                    ),
                    axis=(0, 1)
                ).numpy().tolist()
            )
        ]

    def inference_from_gan_saved_model(
        self, grid_dict_list, gs_image_set, query_images
    ):
        """Inferences GAN SavedModel and gets outputs.

        Args:
            grid_dict_list: list, contains dicts of coordinates and 4-ary tree
                grid indices.
            gs_image_set: set, image output types that are grayscale, i.e.
                have only one channel.
            query_images: tensor, query image tensor of shape
                (batch, patch_height, patch_width, 3).

        Returns:
            image_dict_list: list, dictionaries containing image tensors.
            confusion_matrix_list: list, lists of dictionaries containing
                confusion matrix metrics at thresholds.
        """
        batch_size = len(grid_dict_list)
        image_dict_list = [{} for _ in range(batch_size)]
        confusion_matrix_list = [[] for _ in range(batch_size)]
        image_dict = {}
        if (
            "query_images" in self.image_stitch_types_set and
            len(self.image_stitch_types_set) == 1
        ):
            query_images = self.scale_images(images=query_images)
            image_dict = {"query_images": query_images}
        else:
            Z = None
            if (self.generator_architecture == "berg" and
                self.berg_use_Z_inputs):
                Z = tf.random.normal(
                    shape=(
                        query_images.shape[0], self.berg_latent_size
                    ),
                    mean=self.berg_latent_mean,
                    stddev=self.berg_latent_stddev,
                    dtype=tf.float32
                )
            predictions_by_growth = self.plot_all_exports_by_architecture(
                Z=Z,
                query_images=query_images,
                exports_on_gcs=True,
                export_start_idx=0,
                export_end_idx=1,
                max_size=1024,
                only_output_growth_set={i for i in range(9)},
                num_rows=1,
                generator_architecture=self.generator_architecture,
                overrides={
                    "output_dir": self.gan_export_dir,

                    "export_all_growth_phases": False,

                    "export_query_mahalanobis_distances": True,
                    "export_query_mahalanobis_distance_images_sigmoid": True,
                    "export_query_mahalanobis_distance_images_linear": True,

                    "export_query_pixel_anomaly_flag_images": True,

                    "output_query_images": False,

                    "output_query_gen_encoded_logits": False,
                    "output_query_gen_encoded_images": False,

                    "output_query_enc_encoded_logits": False,
                    "output_query_enc_encoded_images": False,

                    "output_query_anomaly_images_sigmoid": False,
                    "output_query_anomaly_images_linear": False,

                    "output_query_mahalanobis_distances": False,
                    "output_query_mahalanobis_distance_images_sigmoid": False,
                    "output_query_mahalanobis_distance_images_linear": False,

                    "output_query_pixel_anomaly_flag_images": False,

                    "output_query_anomaly_scores": False,
                    "output_query_anomaly_flags": False
                }
            )
            # berg.
            if ("generated_images" in self.image_stitch_types_set and
                self.generator_architecture == "berg"):
                image_dict["generated_images"] = (
                    predictions_by_growth[-1]["generated_images"]
                )
            if ("encoded_generated_images" in self.image_stitch_types_set and
                self.generator_architecture == "berg"):
                image_dict["encoded_generated_images"] = (
                    predictions_by_growth[-1]["encoded_generated_images"]
                )
            if ("query_encoded_images" in self.image_stitch_types_set and
                self.generator_architecture == "berg"):
                image_dict["query_encoded_images"] = (
                    predictions_by_growth[-1]["query_encoded_images"]
                )
            # GANomaly.
            if ("query_gen_encoded_images" in self.image_stitch_types_set and
                self.generator_architecture == "GANomaly"):
                image_dict["query_gen_encoded_images"] = (
                    predictions_by_growth[-1]["query_gen_encoded_images"]
                )
            # berg and GANomaly.
            if "query_images" in self.image_stitch_types_set:
                image_dict["query_images"] = (
                    predictions_by_growth[-1]["query_images"]
                )
            if "query_gen_encoded_images" in self.image_stitch_types_set:
                image_dict["query_gen_encoded_images"] = (
                    predictions_by_growth[-1]["query_gen_encoded_images"]
                )
            if any(
                [
                    "query_anomaly_images_linear_rgb" in self.image_stitch_types_set,
                    "query_anomaly_images_linear_gs" in self.image_stitch_types_set
                ]
            ):
                error_images = predictions_by_growth[-1][
                    "query_anomaly_images_linear"]
                if "query_anomaly_images_linear_rgb" in self.image_stitch_types_set:
                    image_dict["query_anomaly_images_linear_rgb"] = (
                        error_images
                    )
                if "query_anomaly_images_linear_gs" in self.image_stitch_types_set:
                    image_dict["query_anomaly_images_linear_gs"] = (
                        tf.math.reduce_mean(
                            input_tensor=error_images,
                            axis=-1,
                            keepdims=True
                        )
                    )

            if "query_mahalanobis_distance_images_linear" in self.image_stitch_types_set:
                image_dict["query_mahalanobis_distance_images_linear"] = (
                    predictions_by_growth[-1][
                        "query_mahalanobis_distance_images_linear"]
                )
            if any(
                [
                    "query_pixel_anomaly_flag_images" in self.image_stitch_types_set,
                    "kde_rgb" in self.image_stitch_types_set,
                    "kde_gs" in self.image_stitch_types_set,
                    "kde_gs_thresholded" in self.image_stitch_types_set,
                    self.num_confusion_matrix_thresholds > 0,
                    self.custom_mahalanobis_distance_threshold >= 0.0
                ]
            ):
                if self.custom_mahalanobis_distance_threshold >= 0.0:
                    # shape = (batch, height, width)
                    distances = predictions_by_growth[-1][
                        "query_mahalanobis_distances"]
                    flag_images = tf.where(
                        condition=tf.greater(
                            x=distances,
                            y=self.custom_mahalanobis_distance_threshold
                        ),
                        x=tf.ones_like(input=distances, dtype=tf.float32),
                        y=-tf.ones_like(input=distances, dtype=tf.float32)
                    )
                else:
                    # shape = (batch, height, width)
                    flag_images = predictions_by_growth[-1][
                        "query_pixel_anomaly_flag_images"]

                if "query_pixel_anomaly_flag_images" in self.image_stitch_types_set:
                    # shape = (batch, height, width, 1)
                    image_dict["query_pixel_anomaly_flag_images"] = (
                        # shape = (batch, height, width)
                        tf.expand_dims(input=flag_images, axis=-1)
                    )
                if any(
                    [
                        "kde_rgb" in self.image_stitch_types_set,
                        "kde_gs" in self.image_stitch_types_set,
                        "kde_gs_thresholded" in self.image_stitch_types_set,
                        self.num_confusion_matrix_thresholds > 0
                    ]
                ):
                    # rgb shape = (batch, height, width, 3)
                    # gs shape = (batch, height, width, 1)
                    kde_rgb_images, kde_gs_images = (
                        self.get_kernel_density_estimates(
                            anomaly_flags=flag_images,
                            depth=self.patch_depth,
                            bandwidth=self.bandwidth,
                            kernel=self.kernel,
                            metric=self.metric,
                            xbins=self.xbins,
                            ybins=self.ybins,
                            min_neighborhood_count=(
                                self.min_neighborhood_count
                            ),
                            connectivity=self.connectivity,
                            min_anomaly_points_remaining=(
                                self.min_anomaly_points_remaining
                            ),
                            scaling_power=self.scaling_power,
                            scaling_factor=self.scaling_factor,
                            cmap_str=self.cmap_str,
                            dynamic_bandwidth_scale_factor=(
                                self.dynamic_bandwidth_scale_factor
                            ),
                            max_anomaly_points_for_kde=(
                                self.max_anomaly_points_for_kde
                            )
                        )
                    )

                    if "kde_rgb" in self.image_stitch_types_set:
                        image_dict["kde_rgb"] = kde_rgb_images
                    if "kde_gs" in self.image_stitch_types_set:
                        image_dict["kde_gs"] = kde_gs_images
                    if any(
                        [
                            "kde_gs_thresholded" in self.image_stitch_types_set,
                            self.num_confusion_matrix_thresholds > 0
                        ]
                    ):
                        kde_gs_images = (kde_gs_images + 1.) / 2.
                        image_dict["kde_gs_thresholded"] = tf.cast(
                            x=kde_gs_images > self.kde_threshold,
                            dtype=tf.float32
                        ) * 2. - 1.

            if ("annotations" in self.image_stitch_types_set or
                self.num_confusion_matrix_thresholds > 0
               ):
                label_images = []
                for grid_dict in grid_dict_list:
                    corner_width, corner_height = grid_dict["coords"]
                    label_filename = "{}_x_{}_y_{}_width_{}_height_{}_label.png".format(
                        self.annotation_patch_gcs_filepath,
                        corner_width,
                        corner_height,
                        self.patch_width,
                        self.patch_height
                    )
                    if tf.io.gfile.exists(path=label_filename):
                        label_raw_image = tf.io.read_file(filename=label_filename)
                        label_image = tf.io.decode_png(
                            contents=label_raw_image, channels=1
                        )
                    else:
                        label_image = tf.zeros(
                            shape=(self.patch_height, self.patch_width, 1),
                            dtype=tf.uint8
                        )
                    label_images.append(label_image)

                if "annotations" in self.image_stitch_types_set:
                    image_dict["annotations"] = self.scale_images(
                        images=tf.stack(values=label_images, axis=0) * 255
                    )
                if self.num_confusion_matrix_thresholds > 0:
                    for i, label_image in enumerate(label_images):
                        confusion_matrix = self.confusion_matrix(
                            filename=label_filename,
                            image=kde_gs_image,
                            label_image=label_image
                        )
                    confusion_matrix_list[i] = confusion_matrix

            for stitch_type in gs_image_set.intersection(self.image_stitch_types_set):
                image_dict[stitch_type] *= -1

        # Fill image dict list with unstacked tensors for each key.
        for k, v in image_dict.items():
            image_list = tf.unstack(value=v, num=batch_size, axis=0)
            for i in range(batch_size):
                image_dict_list[i][k] = image_list[i]

        return image_dict_list, confusion_matrix_list

    ##########################################################################
    ### Segmentation #########################################################
    ##########################################################################

    class RestoredModel(object):
        """Class restores trained model and performs inference through sess.

        Attributes:
            graph: `tf.Graph`, holds the TensorFlow execution graph.
            sess: `tf.Session`, session to execute TensorFlow graph within.
            model_saver: `tf.compat.v1.train.Saver`, restores trained model
                into session.
            sample_in: tensor, input tensor of segmentation graph.
            c_mask_out: tensor, output tensor of segmentation graph.
        """
        def __init__(self, model_name, model_folder):
            """Constructor of `RestoredModel`.

            Args:
                model_name: str, the full name of the segmentation model.
                model_folder: str, the filepath of the segmentation model.
            """
            self.graph = tf.Graph()
            self.sess = tf.compat.v1.Session(graph=self.graph)

            with self.graph.as_default():
                self.model_saver = tf.compat.v1.train.import_meta_graph(
                    meta_graph_or_file=model_name
                )
                self.model_saver.restore(
                    sess=self.sess,
                    save_path=tf.compat.v1.train.latest_checkpoint(
                        checkpoint_dir=model_folder
                    )
                )
                self.sample_in = self.graph.get_tensor_by_name(
                    name="sample:0"
                )
                self.c_mask_out = self.graph.get_tensor_by_name(
                    name="c_mask:0"
                )

        def run_sess(self, patches):
            """Runs patches through restored trained model session.

            Args:
                patches: np.array of image patches of shape
                    (min(remaining, group_size), patch_size, patch_size, num_channels).

            Returns:
                np.array of segmented iamge of shape
                    (min(remaining, group_size), patch_size, patch_size, 1).
            """
            feed_dict = {self.sample_in: patches}
            generated_mask = self.sess.run(
                fetches=[self.c_mask_out], feed_dict=feed_dict
            )
            return generated_mask

        def close_sess(self):
            """Closes TensorFlow session."""
            self.sess.close()


    def image2patch(
        self,
        in_image,
        patch_size, stride,
        median_blur_image=False,
        median_blur_kernel_size=9
    ):
        """Converts input image to a list of patches.

        Args:
            in_image: tensor, image tensor of shape (height, width, 3).
            patch_size: int, the size of each square patch.
            stride: int, the number of pixels to jump for the next patch.
            median_blur_image: bool, whether to apply `medianBlur` to the
                input image.
            median_blur_kernel_size: int, the kernel size for `medianBlur`.

        Returns:
            np.array of image patches of shape
                (patch_size, patch_size, num_channels).
        """
        if median_blur_image is True:
            in_image = cv2.medianBlur(in_image, median_blur_kernel_size)
        shape = in_image.shape
        if shape[0] < patch_size:
            H = 0
        else:
            H = math.ceil((shape[0] - patch_size) / stride)
        if shape[1] < patch_size:
            W = 0
        else:
            W = math.ceil((shape[1] - patch_size) / stride)
        patch_list = []

        hpad = patch_size + stride * H - shape[0]
        wpad = patch_size + stride * W - shape[1]
        if len(shape) > 2:
            full_image = np.pad(
                in_image, ((0, hpad), (0, wpad), (0, 0)), mode='symmetric'
            )
        else:
            full_image = np.pad(
                in_image, ((0, hpad), (0, wpad)), mode='symmetric'
            )
        for i in range(H + 1):
            hs = i * stride
            he = i * stride + patch_size
            for j in range(W + 1):
                ws = j * stride
                we = j * stride + patch_size
                if len(shape) > 2:
                    # element.shape = (patch_size, patch_size, 3)
                    patch_list.append(full_image[hs:he, ws:we, :])
                else:
                    # element.shape = (patch_size, patch_size)
                    patch_list.append(full_image[hs:he, ws:we])
        if len(patch_list) != (H + 1) * (W + 1):
            raise ValueError('Patch_list: ', str(len(patch_list), ' H: ', str(H), ' W: ', str(W)))
        # len = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1)
        return patch_list


    def list2batch(self, patches):
        """Converts list of patches to a batch of patches.

        Args:
            patches: list, image patches of shape
                (patch_height, patch_width, 3).

        Returns:
            np.array of batch of image patches of shape
                (min(remaining, group_size), patch_size, patch_size, num_channels).
        """
        # (patch_size, patch_size, num_channels).
        patch_shape = list(patches[0].shape)
        # min(remaining, group_size).
        batch_size = len(patches)
        if len(patch_shape) > 2:
            batch = np.zeros([batch_size] + patch_shape)
            for index, temp in enumerate(patches):
                # shape = (min(remaining, group_size), patch_size, patch_size, num_channels).
                batch[index, :, :, :] = temp
        else:
            batch = np.zeros([batch_size] + patch_shape + [1])
            for index, temp in enumerate(patches):
                # shape = (min(remaining, group_size), patch_size, patch_size, 1).
                batch[index, :, :, :] = np.expand_dims(temp, axis=-1)
        return batch


    def preprocess(self, input_image):
        """Preprocesses input image and batches images.

        Args:
            input_image: tensor, image tensor of shape
                (patch_height, patch_width, 3).

        Returns:
            List of length num_group of patch image arrays of shape
                (min(remaining, group_size), patch_size, patch_size, num_channels).
        """
        # len = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1)
        # Each element has shape = (patch_size, patch_size, num_channels).
        patch_list = self.image2patch(
            in_image=tf.cast(x=input_image, dtype=tf.float32) / 255.0,
            patch_size=self.segmentation_patch_size,
            stride=self.segmentation_stride,
            median_blur_image=self.segmentation_median_blur_image,
            median_blur_kernel_size=self.segmentation_median_blur_kernel_size
        )
        # (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1) / group_size
        num_group = math.ceil(len(patch_list) / self.segmentation_group_size)
        batch_group = []
        for i in range(num_group):
            start_idx = i * self.segmentation_group_size
            end_idx = (i + 1) * self.segmentation_group_size
            # shape = (min(remaining, group_size), patch_size, patch_size, num_channels).
            temp_batch = self.list2batch(patch_list[start_idx: end_idx])
            batch_group.append(temp_batch)
        return batch_group


    def batch2list(self, batch):
        """Converts a batch of patches into a list of batches.

        Args:
            restored_model: Restored TensorFlow segmentation model.
            batch: np.array of shape
                (min(remaining, group_size), patch_size, patch_size, num_channels).

        Returns:
            List of length batch.shape[0] of patch image arrays of shape
                (patch_size, patch_size).
        """
        return [batch[index, :, :] for index in range(batch.shape[0])]


    def sess_inference(self, restored_model, batch_group):
        """Inferences model session for each patch in batch group.

        Args:
            restored_model: Restored TensorFlow segmentation model.
            batch_group: list, length of num_group, contains batches of
                patches of shape
                (min(remaining, group_size), patch_size, patch_size, num_channels).

        Returns:
            List of segmented patches.
        """
        patch_list = []
        # len(batch_group) = num_group = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1) / group_size
        # temp_batch.shape = (min(remaining, group_size), patch_size, patch_size, num_channels).
        for temp_batch in batch_group:
            # shape = (min(remaining, group_size), patch_size, patch_size, 1).
            segmented_mask_batch = restored_model.run_sess(temp_batch)[0]
            # shape = (min(remaining, group_size), patch_size, patch_size).
            segmented_mask_batch = np.squeeze(segmented_mask_batch, axis=-1)
            # len(segmented_mask_list) = min(remaining, group_size)
            segmented_mask_list = self.batch2list(segmented_mask_batch)
            patch_list += segmented_mask_list
        # len(patch_list) = num_group = (math.ceil((shape[0] - patch_size) / stride) + 1) * (math.ceil((shape[1] - patch_size) / stride) + 1) / group_size
        return patch_list


    def patch2image(self, patch_list, patch_size, stride, shape):
        """Combines patches from image back into full image.

        Args:
            patch_list: list, patch np.arrays of shape
                (patch_size, patch_size).
            patch_size: int, the size of each square patch.
            stride: int, the number of pixels to jump for the next patch.
            shape: tuple, the shape of the original image.

        Returns:
            np.array of combined patches into a single image.
        """
        if shape[0] < patch_size:
            H = 0
        else:
            H = math.ceil((shape[0] - patch_size) / stride)
        if shape[1] < patch_size:
            W = 0
        else:
            W = math.ceil((shape[1] - patch_size) / stride)

        # shape = (height, width).
        full_image = np.zeros([H * stride + patch_size, W * stride + patch_size])
        # shape = (height, width).
        bk = np.zeros([H * stride + patch_size, W * stride + patch_size])
        cnt = 0
        for i in range(H + 1):
            hs = i * stride
            he = hs + patch_size
            for j in range(W + 1):
                ws = j * stride
                we = ws + patch_size
                full_image[hs:he, ws:we] += patch_list[cnt]
                bk[hs:he, ws:we] += np.ones([patch_size, patch_size])
                cnt += 1
        full_image /= bk
        # numpy array shape = (height, width).
        image = full_image[0:shape[0], 0:shape[1]]
        return image


    def center_point(self, mask):
        """Draws center point of segmentation mask.

        Args:
            mask: tensor, segmentation mask tensor of shape
                (patch_height, patch_width, 1).

        Returns:
            np.array center point of cell segmentation mask.
        """
        import skimage.measure
        import skimage.morphology
        v, h = mask.shape
        center_mask = np.zeros([v, h])
        mask = skimage.morphology.erosion(mask, skimage.morphology.square(3))
        individual_mask = skimage.measure.label(mask, connectivity=2)
        prop = skimage.measure.regionprops(individual_mask)
        for cordinates in prop:
            temp_center = cordinates.centroid
            if not math.isnan(temp_center[0]) and not math.isnan(temp_center[1]):
                temp_mask = np.zeros([v, h])
                temp_mask[int(temp_center[0]), int(temp_center[1])] = 1
                center_mask += skimage.morphology.dilation(
                    temp_mask, skimage.morphology.square(2)
                )
        return np.clip(center_mask, a_min=0, a_max=1).astype(np.uint8)


    def draw_individual_edge(self, mask):
        """Draws individual edge from segmentation mask.

        Args:
            mask: tensor, segmentation mask tensor of shape
                (patch_height, patch_width, 1).

        Returns:
            np.array edge of cell segmentation mask.
        """
        import skimage.measure
        import skimage.morphology
        v, h = mask.shape
        edge = np.zeros([v, h])
        individual_mask = skimage.measure.label(mask, connectivity=2)
        for index in np.unique(individual_mask):
            if index == 0:
                continue
            temp_mask = np.copy(individual_mask)
            temp_mask[temp_mask != index] = 0
            temp_mask[temp_mask == index] = 1
            temp_mask = skimage.morphology.dilation(
                temp_mask, skimage.morphology.square(3)
            )
            temp_edge = cv2.Canny(temp_mask.astype(np.uint8), 2, 5) / 255
            edge += temp_edge
        return np.clip(edge, a_min=0, a_max=1).astype(np.uint8)

    def center_edge(self, mask, image):
        """Calculates centers and edges of cells from segmentation mask.

        Args:
            mask: tensor, segmentation mask tensor of shape
                (patch_height, patch_width, 1).
            image: tensor, image tensor of shape
                (patch_height, patch_width, 3).

        Returns:
            center_edge_masks: np.array, center and edge masks overlaid on
                original image.
            grayscale_maps: np.array, grayscale maps of centers and edges of
                cells.
        """
        # shape = (height, width).
        center_map = self.center_point(mask)
        # shape = (height, width).
        edge_map = self.draw_individual_edge(mask)
        # shape = (height, width).
        comb_mask = center_map + edge_map
        # shape = (height, width).
        comb_mask = np.clip(comb_mask, a_min=0, a_max=1)
        check_image = np.copy(image)
        comb_mask *= 255
        # shape = (height, width, num_channels).
        check_image[:, :, 1] = np.maximum(check_image[:, :, 1], comb_mask)
        return check_image.astype(np.uint8), comb_mask.astype(np.uint8)

    def cell_seg_main(self, query_images):
        """Performs cell segmentation.

        Args:
            query_images: tensor, query image tensor of shape
                (batch, patch_height, patch_width, 3).

        Returns:
            center_edge_masks: list, center and edge masks overlaid on
                original image.
            grayscale_maps: list, grayscale maps of centers and edges of
                cells.
        """
        restored_model = self.RestoredModel(
            model_name=os.path.join(
                self.segmentation_export_dir, self.segmentation_model_name
            ),
            model_folder=self.segmentation_export_dir
        )
        center_edge_masks = []
        gray_maps = []
        for i, query_image in enumerate(query_images):
            batch_group = self.preprocess(query_image)
            mask_list = self.sess_inference(restored_model, batch_group)
            c_mask = self.patch2image(
                patch_list=mask_list,
                patch_size=self.segmentation_patch_size,
                stride=self.segmentation_stride,
                shape=query_image.shape
            )
            c_mask = cv2.medianBlur((255 * c_mask).astype(np.uint8), 3)
            c_mask = c_mask.astype(np.float) / 255
            thr = 0.5
            c_mask[c_mask < thr] = 0
            c_mask[c_mask >= thr] = 1
            center_edge_mask, gray_map = self.center_edge(c_mask, query_image)
            center_edge_masks.append(center_edge_mask)
            gray_maps.append(gray_map)

        restored_model.close_sess()
        # Lengths = query_image.shape[0], number of query images.
        return center_edge_masks, gray_maps

    def mask_to_polygons(self, mask, epsilon=10., min_area=10.):
        """Converts mask to polygons.

        Args:
            mask: tensor, query image tensor of shape
                (batch, patch_height, patch_width, 3).
            min_area: float, minimum amount of area within contour to be added as
                a polygon.

        Returns:
            `MultiPolygon` representation of input mask.
        """
        import shapely.geometry
        # First, find contours with cv2: it's much faster than shapely
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return shapely.geometry.MultiPolygon()
        # Now messy stuff to associate parent and child contours.
        cnt_children = defaultdict(list)
        child_contours = set()
        assert hierarchy.shape[0] == 1
        # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
        for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
            if parent_idx != -1:
                child_contours.add(idx)
                cnt_children[parent_idx].append(contours[idx])
        # Create actual polygons filtering by area (removes artifacts).
        all_polygons = []
        for idx, cnt in enumerate(contours):
            if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
                assert cnt.shape[1] == 1
                poly = shapely.geometry.Polygon(
                    shell=cnt[:, 0, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= min_area])
                all_polygons.append(poly)
        all_polygons = shapely.geometry.MultiPolygon(all_polygons)

        return all_polygons

    def get_cell_polygon_coords(self, query_images):
        """Gets cell polygon coordinates.

        Args:
            query_images: tensor, query image tensor of shape
                (batch, patch_height, patch_width, 3).

        Returns:
            cell_coords_list: list, dictionaries containing lists of cell
                polygon exterior coordinates.
            nuclei_coords_list: list, dictionaries containing lists of cell
                polygon centroid coordinates.
        """
        _, grayscale_maps = self.cell_seg_main(query_images)
        cell_coords_list = []
        nuclei_coords_list = []
        for grayscale_map in grayscale_maps:
            cell_polygons = self.mask_to_polygons(grayscale_map)

            cell_coords_list.append(
                [list(polygon.exterior.coords) for polygon in cell_polygons]
            )
            nuclei_coords_list.append(
                [
                    list(polygon.centroid.coords)[0]
                    for polygon in cell_polygons
                ]
            )

        return cell_coords_list, nuclei_coords_list

    def inference_from_segmentation_saved_model(
        self, grid_dict_list, query_images
    ):
        """Inferences segmentation SavedModel and gets outputs.

        Args:
            grid_dict_list: list, contains dicts of coordinates and 4-ary tree
                grid indices.
            query_images: tensor, query image tensor of shape
                (batch, patch_height, patch_width, 3).

        Returns:
            cell_coords_list: list, dictionaries containing lists of cell
                polygon exterior coordinates.
            nuclei_coords_list: list, dictionaries containing lists of cell
                polygon centroid coordinates.
        """
        cell_coords_list, nuclei_coords_list = self.get_cell_polygon_coords(
            query_images
        )
        cell_coords_dict_list = []
        nuclei_coords_dict_list = []
        for i, grid_dict in enumerate(grid_dict_list):
            cell_coords_dict_list.append(
                {str(grid_dict["coords"]): cell_coords_list[i]}
            )
            nuclei_coords_dict_list.append(
                {str(grid_dict["coords"]): nuclei_coords_list[i]}
            )

        return cell_coords_dict_list, nuclei_coords_dict_list

    def process_patch_grid_elements(self, grid_dict_list, gs_image_set):
        """Processes patch grid elements.

        Args:
            grid_dict_list: list, contains dicts of coordinates and 4-ary tree
                grid indices.
            gs_image_set: set, image output types that are grayscale, i.e.
                have only one channel.

        Returns:
            image_dict_list: list, dictionaries containing image tensors.
            confusion_matrix_list: list, lists of dictionaries containing
                confusion matrix metrics at thresholds.
            cell_coords_list: list, dictionaries containing lists of cell
                polygon exterior coordinates.
            nuclei_coords_list: list, dictionaries containing lists of cell
                polygon centroid coordinates.
            patch_coord_list: list, 2-tuples of patch coordinates.
            
        """
        batch_size = len(grid_dict_list)
        image_dict_list = [{} for _ in range(batch_size)]
        confusion_matrix_list = [[] for _ in range(batch_size)]
        cell_coords_dict_list = [{} for _ in range(batch_size)]
        nuclei_coords_dict_list = [{} for _ in range(batch_size)]

        query_images = self.get_query_images(grid_dict_list)

        if not query_images:
            image_dict_list = self.process_non_patch_grid_elements(
                grid_dict_list, gs_image_set
            )
        else:
            query_images = tf.stack(values=query_images, axis=0)

            if self.image_stitch_types_set:
                (image_dict_list,
                 confusion_matrix_list) = self.inference_from_gan_saved_model(
                    grid_dict_list, gs_image_set, query_images
                )

            if self.segmentation_coord_types_set:
                (cell_coords_dict_list,
                 nuclei_coords_dict_list) = (
                    self.inference_from_segmentation_saved_model(
                        grid_dict_list, query_images
                    )
                )
        return (
            image_dict_list,
            confusion_matrix_list,
            cell_coords_dict_list,
            nuclei_coords_dict_list
        )

    def process(self, grid_element):
        """Processes grid element performing inference in ParDo.

        Args:
            grid_element: 2-tuple, contains batch index and list of dicts of
                4-ary tree grid indices and possibly coordinates of patch
                regions to include in collection.

        Yields:
            Dictionary containing image dictionary, confusion matrix metrics at
                thresholds, and 4-ary tree grid index lists.
        """
        batch_idx, grid_dict_list = grid_element
        # Coerce to list from _UnwindowedValues.
        grid_dict_list = list(grid_dict_list)

        gs_image_set = set(
            [
                "query_anomaly_images_linear_gs",
                "query_mahalanobis_distance_images_linear",
                "query_pixel_anomaly_flag_images",
                "kde_gs",
                "kde_gs_thresholded",
                "annotations"
            ]
        )
        batch_size = len(grid_dict_list)
        image_dict_list = [{} for _ in range(batch_size)]
        confusion_matrix_list = [[] for _ in range(batch_size)]
        cell_coords_dict_list = [{} for _ in range(batch_size)]
        nuclei_coords_dict_list = [{} for _ in range(batch_size)]

        if batch_idx < 0:
            image_dict_list = self.process_non_patch_grid_elements(
                grid_dict_list, gs_image_set
            )
        else:
            (image_dict_list,
             confusion_matrix_list,
             cell_coords_dict_list,
             nuclei_coords_dict_list) = (
                self.process_patch_grid_elements(
                    grid_dict_list, gs_image_set
                )
            )

        for i, grid_dict in enumerate(grid_dict_list):
            yield {
                "images": image_dict_list[i],
                "confusion_matrix": confusion_matrix_list[i],
                "segmentation_cell_coords": cell_coords_dict_list[i],
                "segmentation_nuclei_coords": nuclei_coords_dict_list[i],
                "grid_global_idx_stack": grid_dict["grid_global_idx_stack"],
                "grid_local_idx_stack": grid_dict["grid_local_idx_stack"]
            }
