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

from google.cloud import storage
import numpy as np
import os
import tensorflow as tf

from . import get_predictions
from . import image_utils
from . import kde


def list_gcs_savedmodel_exports(bucket_name, prefix):
    """Lists all the SavedModel exports in bucket in lexigraphical order.

    Args:
        bucket_name: str, name of GCS bucket.
        prefix: str, prefix of GCS bucket blob object.

    Returns:
        List of exported SavedModels within GCS location in lexigraphical
            order.
    """
    storage_client = storage.Client()

    blobs = storage_client.list_blobs(
        bucket_or_name=bucket_name, prefix=prefix
    )

    prefix_len = len(prefix.split("/"))
    export_set = set()
    export_list = []

    for blob in blobs:
        path_split = blob.name.split("/")[0:prefix_len + 1]
        if path_split[-1]:
            if path_split[-1] not in export_set:
                export_list.append(path_split[-1])
                export_set.add(path_split[-1])

    return export_list


def plot_predictions(predictions, num_rows, params):
    """Plots predictions based on bool conditions.

    Args:
        predictions: dict, predictions from exported SavedModel.
        num_rows: int, number of rows to plot for each desired output.
        params: dict, user passed parameters.
    """
    # Using Z.
    if params["export_Z"] and params["output_Z"]:
        print("Z:")
        Z = predictions["Z"]
        print("Z.shape = {}".format(Z.shape))
        print(Z)

    if (
        params["export_generated_images"] and
        params["output_generated_images"]
    ):
        print("Generated images:")
        generated_images = predictions["generated_images"]
        print("generated_images.shape = {}".format(generated_images.shape))

        image_utils.plot_images(
            images=image_utils.descale_images(images=generated_images),
            depth=params["image_depth"],
            num_rows=num_rows
        )

    if (
        params["export_encoded_generated_logits"] and
        params["output_encoded_generated_logits"]
    ):
        print("Encoded generated logits:")
        encoded_generated_logits = predictions["encoded_generated_logits"]
        print(
            "encoded_generated_logits.shape = {}".format(
                encoded_generated_logits.shape
            )
        )
        print(encoded_generated_logits)

    if (
        params["export_encoded_generated_images"] and
        params["output_encoded_generated_images"]
    ):
        print("Encoded generated images:")
        encoded_generated_images = predictions["encoded_generated_images"]
        print(
            "encoded_generated_images.shape = {}".format(
                encoded_generated_images.shape
            )
        )

        image_utils.plot_images(
            images=image_utils.descale_images(
                images=encoded_generated_images
            ),
            depth=params["image_depth"],
            num_rows=num_rows
        )

    # Using query_images.
    if params["export_query_images"] and params["output_query_images"]:
        print("Query images:")
        query_images = predictions["query_images"]
        print("query_images.shape = {}".format(query_images.shape))

        image_utils.plot_images(
            images=image_utils.descale_images(images=query_images),
            depth=params["image_depth"],
            num_rows=num_rows
        )

    if (
        params["export_query_encoded_logits"] and
        params["output_query_encoded_logits"]
    ):
        print("Encoded query logits:")
        query_encoded_logits = predictions["query_encoded_logits"]
        print(
            "query_encoded_logits.shape = {}".format(
                query_encoded_logits.shape
            )
        )
        print(query_encoded_logits)

    if (
        params["export_query_encoded_images"] and
        params["output_query_encoded_images"]
    ):
        print("Encoded query images:")
        query_encoded_images = predictions["query_encoded_images"]
        print(
            "query_encoded_images.shape = {}".format(
                query_encoded_images.shape
            )
        )

        image_utils.plot_images(
            images=image_utils.descale_images(images=query_encoded_images),
            depth=params["image_depth"],
            num_rows=num_rows
        )

    if (
        params["export_query_gen_encoded_logits"] and
        params["output_query_gen_encoded_logits"]
    ):
        print("Generator encoded query logits:")
        query_gen_encoded_logits = predictions["query_gen_encoded_logits"]
        print(
            "query_gen_encoded_logits.shape = {}".format(
                query_gen_encoded_logits.shape
            )
        )
        print(query_gen_encoded_logits)

    if (
        params["export_query_gen_encoded_images"] and
        params["output_query_gen_encoded_images"]
    ):
        print("Generator encoded query images:")
        query_gen_encoded_images = predictions["query_gen_encoded_images"]
        print(
            "query_gen_encoded_images.shape = {}".format(
                query_gen_encoded_images.shape
            )
        )

        image_utils.plot_images(
            images=image_utils.descale_images(
                images=query_gen_encoded_images
            ),
            depth=params["image_depth"],
            num_rows=num_rows
        )

    if (
        params["export_query_enc_encoded_logits"] and
        params["output_query_enc_encoded_logits"]
    ):
        print("Encoder encoded query logits:")
        query_enc_encoded_logits = predictions["query_enc_encoded_logits"]
        print(
            "query_enc_encoded_logits.shape = {}".format(
                query_enc_encoded_logits.shape
            )
        )
        print(query_enc_encoded_logits)

    if (
        params["export_query_enc_encoded_images"] and
        params["output_query_enc_encoded_images"]
    ):
        print("Encoder encoded query images:")
        query_enc_encoded_images = predictions["query_enc_encoded_images"]
        print(
            "query_enc_encoded_images.shape = {}".format(
                query_enc_encoded_images.shape
            )
        )

        image_utils.plot_images(
            images=image_utils.descale_images(
                images=query_enc_encoded_images
            ),
            depth=params["image_depth"],
            num_rows=num_rows
        )

    if (
        params["export_query_anomaly_images_sigmoid"] and
        params["output_query_anomaly_images_sigmoid"]
    ):
        query_anomaly_images_sigmoid = (
            predictions["query_anomaly_images_sigmoid"]
        )
        if params["image_depth"] > 1:
            print("RGB query anomaly images sigmoid:")
            print(
                "rgb_query_anomaly_images_sigmoid.shape = {}".format(
                    query_anomaly_images_sigmoid.shape
                )
            )
            image_utils.plot_images(
                images=image_utils.descale_images(
                    images=query_anomaly_images_sigmoid
                ),
                depth=params["image_depth"],
                num_rows=num_rows
            )

            greyscale_query_anomaly_images_sigmoid = np.mean(
                query_anomaly_images_sigmoid, axis=-1
            )
        else:
            greyscale_query_anomaly_images_sigmoid = (
                query_anomaly_images_sigmoid
            )

        print("Grayscale query anomaly images:")
        print(
            "greyscale_query_anomaly_images_sigmoid.shape = {}".format(
                greyscale_query_anomaly_images_sigmoid.shape
            )
        )
        image_utils.plot_images(
            images=image_utils.descale_images(
                images=greyscale_query_anomaly_images_sigmoid
            ),
            depth=1,
            num_rows=num_rows
        )

    if (
        params["export_query_anomaly_images_linear"] and
        params["output_query_anomaly_images_linear"]
    ):
        query_anomaly_images_linear = (
            predictions["query_anomaly_images_linear"]
        )
        if params["image_depth"] > 1:
            print("RGB query anomaly images linear:")
            print(
                "rgb_query_anomaly_images_linear.shape = {}".format(
                    query_anomaly_images_linear.shape
                )
            )
            image_utils.plot_images(
                images=image_utils.descale_images(
                    images=query_anomaly_images_linear
                ),
                depth=params["image_depth"],
                num_rows=num_rows
            )

            greyscale_query_anomaly_images_linear = np.mean(
                query_anomaly_images_linear, axis=-1
            )
        else:
            greyscale_query_anomaly_images_linear = query_anomaly_images_linear

        print("Grayscale query anomaly images:")
        print(
            "greyscale_query_anomaly_images_linear.shape = {}".format(
                greyscale_query_anomaly_images_linear.shape
            )
        )
        image_utils.plot_images(
            images=image_utils.descale_images(
                images=greyscale_query_anomaly_images_linear
            ),
            depth=1,
            num_rows=num_rows
        )

    if (
        params["export_query_mahalanobis_distances"] and
        params["output_query_mahalanobis_distances"]
    ):
        print("Query Mahalanobis distances:")
        query_mahalanobis_distances = (
            predictions["query_mahalanobis_distances"]
        )
        print(
            "query_mahalanobis_distances.shape = {}".format(
                query_mahalanobis_distances.shape
            )
        )
        print(query_mahalanobis_distances)

    if (
        params["export_query_mahalanobis_distance_images_sigmoid"] and
        params["output_query_mahalanobis_distance_images_sigmoid"]
    ):
        print("Query Mahalanobis distance images sigmoid:")
        query_mahalanobis_distance_images_sigmoid = (
            predictions["query_mahalanobis_distance_images_sigmoid"]
        )
        print(
            "query_mahalanobis_distance_images_sigmoid.shape = {}".format(
                query_mahalanobis_distance_images_sigmoid.shape
            )
        )
        image_utils.plot_images(
            images=image_utils.descale_images(
                images=query_mahalanobis_distance_images_sigmoid
            ),
            depth=1,
            num_rows=num_rows
        )

    if (
        params["export_query_mahalanobis_distance_images_linear"] and
        params["output_query_mahalanobis_distance_images_linear"]
    ):
        print("Query Mahalanobis distance images linear:")
        query_mahalanobis_distance_images_linear = (
            predictions["query_mahalanobis_distance_images_linear"]
        )
        print(
            "query_mahalanobis_distance_images_linear.shape = {}".format(
                query_mahalanobis_distance_images_linear.shape
            )
        )
        image_utils.plot_images(
            images=image_utils.descale_images(
                images=query_mahalanobis_distance_images_linear
            ),
            depth=1,
            num_rows=num_rows
        )

    if (
        params["export_query_pixel_anomaly_flag_images"] and
        params["output_query_pixel_anomaly_flag_images"]
    ):
        print("Query pixel anomaly flag images:")
        query_pixel_anomaly_flag_images = (
            predictions["query_pixel_anomaly_flag_images"]
        )
        print(
            "query_pixel_anomaly_flag_images.shape = {}".format(
                query_pixel_anomaly_flag_images.shape
            )
        )
        image_utils.plot_images(
            images=image_utils.descale_images(
                images=query_pixel_anomaly_flag_images
            ),
            depth=1,
            num_rows=num_rows
        )

    if (
        params["export_query_pixel_anomaly_flag_counts"] and
        params["output_query_pixel_anomaly_flag_counts"]
    ):
        print("Query pixel anomaly flag counts:")
        query_pixel_anomaly_flag_counts = (
            predictions["query_pixel_anomaly_flag_counts"]
        )
        print(
            "query_pixel_anomaly_flag_counts.shape = {}".format(
                query_pixel_anomaly_flag_counts.shape
            )
        )
        print(query_pixel_anomaly_flag_counts)

    if (
        params["export_query_pixel_anomaly_flag_percentages"] and
        params["output_query_pixel_anomaly_flag_percentages"]
    ):
        print("Query pixel anomaly flag percentages:")
        query_pixel_anomaly_flag_percentages = (
            predictions["query_pixel_anomaly_flag_percentages"]
        )
        print(
            "query_pixel_anomaly_flag_percentages.shape = {}".format(
                query_pixel_anomaly_flag_percentages.shape
            )
        )
        print(query_pixel_anomaly_flag_percentages)

    if all(
        [
            params["generator_architecture"] == "berg",
            params["export_query_anomaly_scores"],
            params["output_query_anomaly_scores"]
        ]
    ):
        print("Query anomaly scores:")
        query_anomaly_scores = predictions["query_anomaly_scores"]
        print(
            "query_anomaly_scores.shape = {}".format(
                query_anomaly_scores.shape
            )
        )
        print(query_anomaly_scores)

    if all(
        [
            params["generator_architecture"] == "berg",
            params["export_query_anomaly_flags"],
            params["output_query_anomaly_flags"]
        ]
    ):
        print("Query anomaly flags:")
        query_anomaly_flags = predictions["query_anomaly_flags"]
        print(
            "query_anomaly_flags.shape = {}".format(query_anomaly_flags.shape)
        )
        print(query_anomaly_flags)


def plot_all_exports(
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
    if exports_on_gcs:
        export_list = list_gcs_savedmodel_exports(
            bucket_name=params["output_dir"].split("/")[2],
            prefix="{}/{}".format(
                "/".join(params["output_dir"].split("/")[3:]), "export"
            )
        )
    else:
        export_list = sorted(
            os.listdir(os.path.join(params["output_dir"], "export"))
        )
    print(
        "There are {} exported models at path {}".format(
            len(export_list),
            os.path.join(params["output_dir"], "export")
        )
    )
    print(export_list)

    assert 0 <= export_start_idx < len(export_list)
    assert 0 < export_end_idx <= len(export_list)

    for i in range(export_start_idx, export_end_idx):
        export = export_list[i]
        print(
            "\nexport_idx = {}, export_dir = {}/export/{}".format(
                i, params["output_dir"], export
            )
        )

        if params["export_all_growth_phases"]:
            predictions_by_growth = get_predictions.get_all_growth_predictions(
                export_name=export,
                Z=Z,
                query_images=image_utils.scale_images(query_images),
                max_size=max_size,
                only_output_growth_set=only_output_growth_set,
                params=params
            )

            if params["output_transition_growths"]:
                if params["output_stable_growths"]:
                    print("Outputting all growths!")
                else:
                    print("Outputting just transition growths!")
                    del predictions_by_growth[0::2]
            elif not params["output_transition_growths"]:
                if params["output_stable_growths"]:
                    print("Outputting just stable growths!")
                    del predictions_by_growth[1::2]
                else:
                    print("Outputting no growths!")
                    break
        else:
            predictions_by_growth = (
                get_predictions.get_current_growth_predictions(
                    export_name=export,
                    Z=Z,
                    query_images=image_utils.scale_images(query_images),
                    params=params
                )
            )

        for i, growth in enumerate(predictions_by_growth):
            if i in only_output_growth_set:
                plot_predictions(
                    predictions=growth, num_rows=num_rows, params=params
                )

    return predictions_by_growth


def plot_all_exports_by_architecture(
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
        "export_query_pixel_anomaly_flag_counts": False,
        "export_query_pixel_anomaly_flag_percentages": False,

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
        "output_query_pixel_anomaly_flag_counts": False,
        "output_query_pixel_anomaly_flag_percentages": False,

        "output_query_anomaly_scores": False,
        "output_query_anomaly_flags": False,

        "output_transition_growths": False,
        "output_stable_growths": True,
        "image_depth": 3
    }

    if generator_architecture == "berg":
        params = {
            "export_Z": True,
            "export_generated_images": True,
            "export_encoded_generated_logits": True,
            "export_encoded_generated_images": True,

            "export_query_encoded_logits": True,
            "export_query_encoded_images": True,

            "export_query_gen_encoded_logits": False,
            "export_query_gen_encoded_images": False,
            "export_query_enc_encoded_logits": False,
            "export_query_enc_encoded_images": False,

            "export_query_anomaly_scores": True,
            "export_query_anomaly_flags": True
        }
    elif generator_architecture == "GANomaly":
        Z = None

        params = {
            "export_Z": False,
            "export_generated_images": False,
            "export_encoded_generated_logits": False,
            "export_encoded_generated_images": False,

            "export_query_encoded_logits": False,
            "export_query_encoded_images": False,

            "export_query_gen_encoded_logits": True,
            "export_query_gen_encoded_images": True,
            "export_query_enc_encoded_logits": True,
            "export_query_enc_encoded_images": True,

            "export_query_anomaly_scores": False,
            "export_query_anomaly_flags": False
        }

    params.update(shared_config)

    for key in overrides.keys():
        if key in params:
            params[key] = overrides[key]

    return plot_all_exports(
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


def build_output_type_set(output_type_list, config):
    """Builds set of output types.

    Args:
        output_type_list: list, possible output image types.
        config: dict, user passed parameters.

    Returns:
        Set of requested output image types.
    """
    output_types = set()
    for output_type in output_type_list:
        if config["output_{}".format(output_type)]:
            output_types.add(output_type)
    return output_types


def inference_from_saved_model(
    query_images, output_types_set, gs_image_set, config
):
    """Inferences from SavedModel and performs KDE smoothing.

    Args:
        query_images: tensor, real images to query the model with of shape
            (batch_size, height, width, num_channels)
        output_types_set: set, model output types requested.
        gs_image_set: set, image output types that are grayscale, i.e. have
            only one channel.
        config: dict, user passed parameters.
    """
    image_dict = {}
    if (
        "query_images" in output_types_set and
        len(output_types_set) == 1
    ):
        images = image_utils.scale_images(images=query_images)
        image_dict = {"query_images": images}
    else:
        Z = None
        if (config["generator_architecture"] == "berg" and
            config["berg_use_Z_inputs"]):
            Z = tf.random.normal(
                shape=(
                    query_images.shape[0], config["berg_latent_size"]
                ),
                mean=config["berg_latent_mean"],
                stddev=config["berg_latent_stddev"],
                dtype=tf.float32
            )
        predictions_by_growth = plot_all_exports_by_architecture(
            Z=Z,
            query_images=query_images,
            exports_on_gcs=config["exports_on_gcs"],
            export_start_idx=0,
            export_end_idx=1,
            max_size=1024,
            only_output_growth_set={i for i in range(9)},
            num_rows=1,
            generator_architecture=config["generator_architecture"],
            overrides={
                "output_dir": config["gan_export_dir"],

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
        if ("generated_images" in output_types_set and
            config["generator_architecture"] == "berg"):
            image_dict["generated_images"] = (
                predictions_by_growth[-1]["generated_images"]
            )
        if ("encoded_generated_images" in output_types_set and
            config["generator_architecture"] == "berg"):
            image_dict["encoded_generated_images"] = (
                predictions_by_growth[-1]["encoded_generated_images"]
            )
        if ("query_encoded_images" in output_types_set and
            config["generator_architecture"] == "berg"):
            image_dict["query_encoded_images"] = (
                predictions_by_growth[-1]["query_encoded_images"]
            )
        # GANomaly.
        if ("query_gen_encoded_images" in output_types_set and
            config["generator_architecture"] == "GANomaly"):
            image_dict["query_gen_encoded_images"] = (
                predictions_by_growth[-1]["query_gen_encoded_images"]
            )
        # berg and GANomaly.
        if "query_images" in output_types_set:
            image_dict["query_images"] = (
                predictions_by_growth[-1]["query_images"]
            )
        if any(
            [
                "query_anomaly_images_linear_rgb" in output_types_set,
                "query_anomaly_images_linear_gs" in output_types_set
            ]
        ):
            image = (
                predictions_by_growth[-1][
                    "query_anomaly_images_linear"]
            )
            if "query_anomaly_images_linear_rgb" in output_types_set:
                image_dict["query_anomaly_images_linear_rgb"] = image
            if "query_anomaly_images_linear_gs" in output_types_set:
                image_dict["query_anomaly_images_linear_gs"] = (
                    tf.math.reduce_mean(
                        image, axis=-1, keepdims=True
                    )
                )
        # berg and GANomaly.
        if "query_mahalanobis_distance_images_linear" in output_types_set:
            image_dict["query_mahalanobis_distance_images_linear"] = (
                predictions_by_growth[-1][
                    "query_mahalanobis_distance_images_linear"]
            )
        # berg and GANomaly.
        if any(
            [
                "query_pixel_anomaly_flag_images" in output_types_set,
                "kde_rgb" in output_types_set,
                "kde_gs" in output_types_set,
                "kde_gs_thresholded" in output_types_set,
                "kde_gs_polygon" in output_types_set,
                config["custom_mahalanobis_distance_threshold"] >= 0.0
            ]
        ):
            if config["custom_mahalanobis_distance_threshold"] >= 0.0:
                # shape = (batch, height, width)
                distances = predictions_by_growth[-1][
                    "query_mahalanobis_distances"]
                image = tf.where(
                    condition=tf.greater(
                        x=distances,
                        y=config["custom_mahalanobis_distance_threshold"]
                    ),
                    x=tf.ones_like(input=distances, dtype=tf.float32),
                    y=-tf.ones_like(input=distances, dtype=tf.float32)
                )
            else:
                # shape = (batch, height, width)
                image = predictions_by_growth[-1][
                    "query_pixel_anomaly_flag_images"]

            if "query_pixel_anomaly_flag_images" in output_types_set:
                # shape = (batch, height, width, 1)
                image_dict["query_pixel_anomaly_flag_images"] = (
                    tf.expand_dims(
                        # shape = (batch, height, width)
                        input=image, axis=-1
                    )
                )
            if any(
                [
                    "kde_rgb" in output_types_set,
                    "kde_gs" in output_types_set,
                    "kde_gs_thresholded" in output_types_set,
                    "kde_gs_polygon" in output_types_set
                ]
            ):
                # rgb shape = (batch, height, width, 3)
                # gs shape = (batch, height, width, 1)
                mesh_rgb, mesh_gs = kde.get_kernel_density_estimates(
                    anomaly_flags=image,
                    depth=config["patch_depth"],
                    bandwidth=config["bandwidth"],
                    kernel=config["kernel"],
                    metric=config["metric"],
                    xbins=config["xbins"],
                    ybins=config["ybins"],
                    min_neighborhood_count=config["min_neighborhood_count"],
                    connectivity=config["connectivity"],
                    min_anomaly_points_remaining=(
                        config["min_anomaly_points_remaining"]
                    ),
                    scaling_power=config["scaling_power"],
                    scaling_factor=config["scaling_factor"],
                    cmap_str=config["cmap_str"]
                )

                if "kde_rgb" in output_types_set:
                    image_dict["kde_rgb"] = mesh_rgb
                if any(
                    [
                        "kde_gs" in output_types_set,
                        "kde_gs_polygon" in output_types_set
                    ]
                ):
                    image_dict["kde_gs"] = mesh_gs
                if any(
                    [
                        "kde_gs_thresholded" in output_types_set
                    ]
                ):
                    kde_gs_image = (mesh_gs + 1.) / 2.
                    image_dict["kde_gs_thresholded"] = tf.cast(
                        x=kde_gs_image > config["kde_threshold"],
                        dtype=tf.float32
                    ) * 2. - 1.

        for stitch_type in gs_image_set.intersection(output_types_set):
            image_dict[stitch_type] *= -1

        image_dict = {
            k: np.rot90(m=v, k=2, axes=(0, 1)) for k, v in image_dict.items()
        }
    return image_dict
