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

import math
import os
import tensorflow as tf


def get_saved_model_serving_signatures(export_name, params):
    """Gets SavedModel's serving signatures for inference.

    Args:
        export_name: str, name of exported SavedModel.
        params: dict, user passed parameters.

    Returns:
        Loaded SavedModel and its serving signatures for inference.
    """
    loaded_model = tf.saved_model.load(
        export_dir=os.path.join(
            params["output_dir"], "export", export_name
        )
    )
    print("signature_keys = {}".format(list(loaded_model.signatures.keys())))

    infer = loaded_model.signatures["serving_default"]
    print("structured_outputs = {}".format(infer.structured_outputs))

    # Loaded model also needs to be returned so that infer can find the
    # variables within the graph in the outer scope.
    return loaded_model, infer

def create_export_bool_lists(params):
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
        params["export_query_pixel_anomaly_flag_counts"],
        params["export_query_pixel_anomaly_flag_percentages"],
        params["export_query_anomaly_scores"],
        params["export_query_anomaly_flags"]
    ]

    return export_Z_bool_list, export_query_image_bool_list

def parse_predictions_dict(predictions, num_growths):
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

def get_current_growth_predictions(export_name, Z, query_images, params):
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
    loaded_model, infer = get_saved_model_serving_signatures(
        export_name, params
    )

    (export_Z_bool_list,
     export_query_image_bool_list) = create_export_bool_lists(params)

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

    predictions_by_growth = parse_predictions_dict(
        predictions=predictions, num_growths=1
    )

    return predictions_by_growth

def get_all_growth_predictions_using_Z_and_query_images_berg(
    Z, query_images, max_size, only_output_growth_set, infer
):
    """Gets predictions for all growths using Z and query images.

    Args:
        Z: tensor, random latent vector of shape
            (batch_size, generator_latent_size).
        query_images: tensor, real images to query the model with of shape
            (batch_size, height, width, num_channels).
        max_size: int, the maximum image size within the exported SavedModel.
        only_output_growth_set: set, whether to output growth block.
        infer: SignatureDef, loaded SavedModel's serving signature def to be
            used for inference.

    Returns:
        Dictionary with exported names for keys and predictions tensors for
            values.
    """
    assert(max_size % 2 == 0)
    num_blocks = int(math.log(max_size, 2)) - 1

    kwargs = {"generator_decoder_inputs": Z}

    kwargs.update(
        {
            "encoder_{0}x{0}_inputs".format(4 * 2 ** i): (
                tf.image.resize(
                    images=(
                        query_images
                        if i in only_output_growth_set
                        else query_images[0:0]
                    ),
                    size=[4 * 2 ** i, 4 * 2 ** i]
                )
            )
            for i in range(num_blocks)
        }
    )

    predictions = infer(**kwargs)

    return predictions

def get_all_growth_predictions_using_query_images_berg(
    query_images, max_size, only_output_growth_set, infer
):
    """Gets predictions for all growths using query images.

    Args:
        query_images: tensor, real images to query the model with of shape
            (batch_size, height, width, num_channels).
        max_size: int, the maximum image size within the exported SavedModel.
        only_output_growth_set: set, whether to output growth block.
        infer: SignatureDef, loaded SavedModel's serving signature def to be
            used for inference.

    Returns:
        Dictionary with exported names for keys and predictions tensors for
            values.
    """
    assert(max_size % 2 == 0)
    num_blocks = int(math.log(max_size, 2)) - 1

    kwargs = {
        "encoder_{0}x{0}_inputs".format(4 * 2 ** i): (
            tf.image.resize(
                images=(
                    query_images
                    if i in only_output_growth_set
                    else query_images[0:0]
                ),
                size=[4 * 2 ** i, 4 * 2 ** i]
            )
        )
        for i in range(num_blocks)
    }

    predictions = infer(**kwargs)

    return predictions

def get_all_growth_predictions_using_query_images_ganomaly(
    query_images, max_size, only_output_growth_set, infer
):
    """Gets predictions for all growths using query images.

    Args:
        query_images: tensor, real images to query the model with of shape
            (batch_size, height, width, num_channels).
        max_size: int, the maximum image size within the exported SavedModel.
        only_output_growth_set: set, whether to output growth block.
        infer: SignatureDef, loaded SavedModel's serving signature def to be
            used for inference.

    Returns:
        Dictionary with exported names for keys and predictions tensors for
            values.
    """
    assert(max_size % 2 == 0)
    num_blocks = int(math.log(max_size, 2)) - 1

    kwargs = {
        "generator_encoder_{0}x{0}_inputs".format(4 * 2 ** i): (
            tf.image.resize(
                images=(
                    query_images
                    if i in only_output_growth_set
                    else query_images[0:0]
                ),
                size=[4 * 2 ** i, 4 * 2 ** i]
            )
        )
        for i in range(num_blocks)
    }

    predictions = infer(**kwargs)

    return predictions

def get_all_growth_predictions(
    export_name, Z, query_images, max_size, only_output_growth_set, params
):
    """Gets predictions for all growths from exported SavedModel.

    Args:
        export_name: str, name of exported SavedModel.
        Z: tensor, random latent vector of shape
            (batch_size, generator_latent_size).
        query_images: tensor, real images to query the model with of shape
            (batch_size, height, width, num_channels).
        max_size: int, the maximum image size within the exported SavedModel.
        only_output_growth_set: set, whether to output growth block.
        params: dict, user passed parameters.

    Returns:
        List of num_growths length of dictionaries with fixed keys and
            predictions.
    """
    loaded_model, infer = get_saved_model_serving_signatures(
        export_name, params
    )

    (export_Z_bool_list,
     export_query_image_bool_list) = create_export_bool_lists(params)

    if params["generator_architecture"] == "berg":
        if Z is not None and any(export_Z_bool_list):
            if query_images is not None and any(export_query_image_bool_list):
                predictions = (
                    get_all_growth_predictions_using_Z_and_query_images_berg(
                        Z, query_images, max_size, only_output_growth_set, infer
                    )
                )
            else:
                predictions = infer(generator_inputs=Z)
        else:
            if query_images is not None and any(export_query_image_bool_list):
                predictions = (
                    get_all_growth_predictions_using_query_images_berg(
                        query_images, max_size, only_output_growth_set, infer
                    )
                )
            else:
                print("Nothing was exported, so nothing to infer.")
    elif params["generator_architecture"] == "GANomaly":
        if query_images is not None and any(export_query_image_bool_list):
            predictions = (
                get_all_growth_predictions_using_query_images_ganomaly(
                    query_images, max_size, only_output_growth_set, infer
                )
            )
        else:
            print("Nothing was exported, so nothing to infer.")

    predictions_by_growth = parse_predictions_dict(
        predictions=predictions,
        num_growths=(int(math.log(max_size, 2)) - 2) * 2 + 1
    )

    return predictions_by_growth
