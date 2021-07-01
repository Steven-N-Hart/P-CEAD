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

import shapely.affinity
import shapely.geometry
import shapely.ops
import skimage.measure
import tensorflow as tf

from proganomaly_modules.inference_module import image_utils


def create_patch_polygon(
    patch_coords,
    patch_height,
    patch_width,
    height_scale_factor,
    width_scale_factor
):
    """Creates polygon using patch corner vertices.

    Args:
        patch_coords: 2-tuple, x, y coordinates.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        height_scale_factor: float, scale factor for height.
        width_scale_factor: float, scale factor for width.

    Returns:
        `Polygon` with patch coordinates for exterior.
    """
    w, h = patch_coords
    w, h = w * width_scale_factor, h * height_scale_factor
    patch_width, patch_height = (
        patch_width * width_scale_factor, patch_height * height_scale_factor
    )
    top_left = shapely.geometry.Point(h, w)
    top_right = shapely.geometry.Point(h, w + patch_width)
    bottom_left = shapely.geometry.Point(h + patch_height, w)
    bottom_right = shapely.geometry.Point(h + patch_height, w + patch_width)
    return shapely.geometry.Polygon(
        shell=[top_left, top_right, bottom_right, bottom_left, top_left]
    )


def create_patch_polygons(
    patch_coords_list,
    patch_height,
    patch_width,
    height_scale_factor,
    width_scale_factor
):
    """Creates polygons everywhere there is a patch from list of coordinates.

    Args:
        patch_coords_list: list, patch x and y coordinate 2-tuples.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        height_scale_factor: float, scale factor for height.
        width_scale_factor: float, scale factor for width.

    Returns:
        `MultiPolygon` encapsulating everywhere there is a patch.
    """
    return shapely.geometry.MultiPolygon(
        polygons=[
            create_patch_polygon(
                patch_coords,
                patch_height,
                patch_width,
                height_scale_factor,
                width_scale_factor
            )
            for patch_coords in patch_coords_list
        ]
    ).buffer(0.0)


def process_image(image):
    """Processes image into correct format.

    Args:
        image: np.array, bitmask image array of shape
            (height, width, 1).

    Returns:
        Processed image tensor in range [0., 1.].
    """
    return tf.cast(
        x=255 - image_utils.descale_images(image), dtype=tf.float32
    ).numpy() / 255.


def get_image_bounds(query_image):
    """Processes image into correct format.

    Args:
        query_image: np.array, query image array of shape
            (height, width, 3).

    Returns:
        Processed image tensor in range [0., 1.].
    """
    image_max_across_channels = tf.math.reduce_max(
        input_tensor=query_image, axis=-1
    )
    used_coords = tf.where(condition=image_max_across_channels != 0.0)
    return tf.math.reduce_max(input_tensor=used_coords, axis=0).numpy()


def create_original_multipolygon(image):
    """Creates original `MultiPolygon` of image.

    Args:
        image: tensor, image tensor in range [0., 1.].

    Returns:
        `MultiPolygon` object of image.
    """
    # Create contours list from the image.
    contours = skimage.measure.find_contours(
        image,
        level=0.5,
        fully_connected="low",
        positive_orientation="low",
        mask=None
    )

    # Convert contours into polygons and rotate.
    multipolygon = shapely.geometry.MultiPolygon(
        polygons=[shapely.geometry.Polygon(contour) for contour in contours]
    )
    if not isinstance(multipolygon, shapely.geometry.MultiPolygon):
        multipolygon = [multipolygon]

    # Union any polygons together to reduce the number.
    multipolygon = shapely.ops.unary_union(geoms=multipolygon)
    if not isinstance(multipolygon, shapely.geometry.MultiPolygon):
        multipolygon = [multipolygon]

    # Convert back into a MultiPolygon.
    multipolygon = shapely.geometry.MultiPolygon(polygons=multipolygon)
    if not isinstance(multipolygon, shapely.geometry.MultiPolygon):
        multipolygon = [multipolygon]
    return multipolygon


def clip_vertex(vertex, x_bounds, y_bounds):
    """Clips `Polygon` vertex within x_bounds and y_bounds.

    Args:
        vertex: 2-tuple, the x and y-coorinates of `Polygon` vertex.
        x_bounds: 2-tuple, the min and max bounds in the x-dimension of the
            original image.
        y_bounds: 2-tuple, the min and max bounds in the y-dimension of the
            original image.

    Returns:
        2-tuple, the clipped x and y-coorinates of `Polygon` vertex.
    """
    x = vertex[0]
    if x < x_bounds[0]:
        x = x_bounds[0]
    elif x > x_bounds[1]:
        x = x_bounds[1]

    y = vertex[1]
    if y < y_bounds[0]:
        y = y_bounds[0]
    elif y > y_bounds[1]:
        y = y_bounds[1]
    return (x, y)


def clip_polygon_vertices(polygon, x_bounds, y_bounds):
    """Clips `Polygon` vertices within x_bounds and y_bounds.

    Args:
        polygon: `Polygon` object defined by vertices.
        x_bounds: 2-tuple, the min and max bounds in the x-dimension of the
            original image.
        y_bounds: 2-tuple, the min and max bounds in the y-dimension of the
            original image.

    Returns:
        `Polygon` object of image with possibly clipped vertices.
    """
    vertices = list(polygon.exterior.coords)
    fixed_vertices = [
        clip_vertex(vertex, x_bounds, y_bounds) for vertex in vertices
    ]
    return shapely.geometry.Polygon(
        [shapely.geometry.Point(x, y) for x, y in fixed_vertices]
    )


def create_prediction_polygons(
    query_image,
    kde_gs_image,
    threshold,
    dilation_factor,
    dilation_origin,
    patch_polygons=None
):
    """Processes threshold and dilation factor to return `MultiPolygon`.

    Args:
        query_image: np.array, query image array of shape
            (height, width, 3).
        kde_gs_image: np.array, bitmask image array of shape
            (height, width, 1).
        threshold: float, threshold between [0., 1.] to create binary mask of
            image.
        dilation_factor: float, factor to scale/dilate the `MultiPolygon`.
        dilation_origin: str, the origin each should polygon be scaled about.
            'center' or 'centroid'.
        patch_polygons: `MultiPolygon`, polygons of patch blocks.

    Returns:
        Dilated `MultiPolygon` of thresholded KDE grayscale image.
    """
    x_max, y_max = get_image_bounds(process_image(query_image))
    x_bounds = (0, x_max)
    y_bounds = (0, y_max)

    kde_gs_image = process_image(kde_gs_image)[:, :, 0]

    kde_gs_thresholded_multipolygon = create_original_multipolygon(
        image=kde_gs_image > threshold
    )

    prediction_multipolygon = shapely.ops.unary_union(
        [
            clip_polygon_vertices(
                polygon=shapely.affinity.scale(
                    geom=polygon,
                    xfact=dilation_factor,
                    yfact=dilation_factor,
                    origin=dilation_origin
                ),
                x_bounds=x_bounds,
                y_bounds=y_bounds
            )
            for polygon in kde_gs_thresholded_multipolygon
        ]
    ).buffer(0.0)

    if patch_polygons is not None:
        prediction_multipolygon = prediction_multipolygon.intersection(
            other=patch_polygons
        )

    return prediction_multipolygon
