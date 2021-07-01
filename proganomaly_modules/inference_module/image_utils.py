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
import shapely.geometry
import tensorflow as tf


def decode_png_image_file(filename, channels):
    """Decodes PNG image files.

    Args:
        filename: str, name of PNG image file.
        channels: int, number of channels of decoded PNG image.

    Returns:
        Tensor of shape (height, width, channels).
    """
    raw_image = tf.io.read_file(filename=filename)
    return tf.io.decode_png(contents=raw_image, channels=channels)


def scale_images(images):
    """Scales images from [0, 255] to [-1., 1.].

    Args:
        images: np.array, array of images with range [0, 255] of shape
            (num_images, height, width, num_channels).
    Returns:
        Tensor of images with range [-1., 1.] of shape
            (num_images, height, width, num_channels).
    """
    return tf.clip_by_value(
        t=tf.cast(x=images, dtype=tf.float32) * (2. / 255) - 1.,
        clip_value_min=-1.,
        clip_value_max=1.
    )


def descale_images(images):
    """Descales images from [-1., 1.] to [0, 255].

    Args:
        images: np.array, array of images with range [-1., 1.] of shape
            (num_images, height, width, num_channels).
    Returns:
        Tensor of images with range [0, 255] of shape
            (num_images, height, width, num_channels).
    """
    return tf.clip_by_value(
        t=tf.cast(
            x=((images + 1.0) * (255. / 2)),
            dtype=tf.int32
        ),
        clip_value_min=0,
        clip_value_max=255
    )


def plot_images(images, depth, num_rows):
    """Plots images of given number of channels for the given number of rows.

    Args:
        images: np.array, array of images of
            [num_images, image_size, image_size, num_channels].
        depth: int, number of channels of image.
        num_rows: int, number of rows of image grid.
    """
    num_images = len(images)

    plt.figure(figsize=(20, 20))
    for i in range(num_images):
        image = images[i]
        if num_images % num_rows == 0:
            plt.subplot(num_rows, num_images // num_rows, i + 1)
        else:
            plt.subplot(num_images, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if depth == 1:
            if len(image.shape) == 4:
                plt.imshow(
                    tf.reshape(image, image.shape[:-1]), cmap="gray_r"
                )
            else:
                plt.imshow(image, cmap="gray_r")
        elif depth == 3:
            plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


def plot_geometry_contours(
    geometry, image, fig_name, exterior_color="b", interior_color="k"
):
    """Plots Shapely geometry contours.

    Args:
        geometry: Shapely geometry, could be a `MultiPolygon`,
            `GeometryCollection`, `Polygon`, or `LineString`.
        image: tensor, image tensor to plot contours on top of.
        fig_name: str, name of figure to save to disk. If blank, then doesn't
            save.
        exterior_color: str, the color code for the contour lines of the
            exteriors of the geometry.
        interior_color: str, the color code for the contour lines of the
            interiors of the geometry.
    """
    descaled_image = descale_images(images=image)
    image_minimized = tf.math.reduce_min(input_tensor=descaled_image, axis=-1)
    non_blank_pixels = tf.where(condition=image_minimized != 255)
    max_h, max_w = tf.math.reduce_max(
        input_tensor=non_blank_pixels, axis=0).numpy()
    fig, ax = plt.subplots(figsize=(20, 20))
    plt.imshow(descaled_image[:max_h, :max_w])
    if isinstance(geometry, shapely.geometry.MultiPolygon):
        for i, polygon in enumerate(geometry):
            y, x = polygon.exterior.xy
            ax.plot(x, y, linewidth=2, color=exterior_color)
            for interior in polygon.interiors:
                y, x = interior.xy
                ax.plot(x, y, linewidth=2, color=interior_color)
    elif isinstance(geometry, shapely.geometry.GeometryCollection):
        geoms = geometry.geoms
        for i, geom in enumerate(geoms):
            if isinstance(geom, shapely.geometry.Polygon):
                y, x = geom.exterior.xy
                ax.plot(x, y, linewidth=2, color=exterior_color)
                for interior in geom.interiors:
                    y, x = interior.xy
                    ax.plot(x, y, linewidth=2, color=interior_color)
            else:
                y, x = geom.xy
                ax.plot(x, y, linewidth=2, color=exterior_color)    
    elif isinstance(geometry, shapely.geometry.Polygon):
        if shapely.geometry.mapping(geometry)["coordinates"]:
            y, x = geometry.exterior.xy
            ax.plot(x, y, linewidth=2, color=exterior_color)
            for interior in geometry.interiors:
                y, x = interior.xy
                ax.plot(x, y, linewidth=2, color=interior_color)
    elif isinstance(geometry, shapely.geometry.LineString):
        y, x = multipolygon.xy
        ax.plot(x, y, linewidth=2, color=exterior_color)
        
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(left=0, right=max_w)
    plt.ylim(bottom=max_h, top=0)
    plt.axis("off")
    if fig_name:
        plt.savefig("{}.png".format(fig_name), dpi=fig.dpi, bbox_inches="tight", pad_inches=0.0)
    fig.show()