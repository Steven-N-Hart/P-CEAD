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

import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import sklearn.neighbors
import tensorflow as tf


def kde2D(x, y, bandwidth, kernel, metric, xbins, ybins, **kwargs):
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
        kwargs: dict, any other keyword args to pass to
            sklearn.neighbors.KernelDensity.

    Returns:
        np.array of the log-likelihood of samples of shape (xbins, ybins).
    """
    xbins = complex(0, xbins)
    ybins = complex(0, ybins)

    # Create grid of sample locations.
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

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
    cmap_str
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
        connectivity: int, connectivity defining the neighborhood of a pixel.
        min_anomaly_points_remaining: minimum number of anomaly points
            following removing small objects to not clear all flags.
        scaling_power: float, the exponent to use for scaling.
        scaling_factor: float, positive factor to scale anomaly flag
            counts by.
        cmap_str: str, which color map to use.

    Returns:
        mesh_rgb: np.array, RGB KDE image of shape
            (batch_size, height, width, 3).
        mesh_gs: np.array, grayscale KDE image of shape
            (batch_size, height, width, 1).
    """
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
            # each shape = (num_true,).
            x, y = anomaly_points[:, 0], anomaly_points[:, 1]

            # shape = (xbins, ybins).
            zz = kde2D(
                x=x,
                y=y,
                bandwidth=bandwidth,
                kernel=kernel,
                metric=metric,
                xbins=xbins,
                ybins=ybins
            )
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