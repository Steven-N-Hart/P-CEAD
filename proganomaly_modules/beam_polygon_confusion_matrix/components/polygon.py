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

import apache_beam as beam
import tensorflow as tf


class PolygonDoFn(beam.DoFn):
    """ParDo class that creates KDE & annotation polygon confusion matrix.

    Attributes:
        slide_name: str, name of slide.
        annotations_image_gcs_path: str, GCS path to annotations image.
        kde_gs_image_gcs_path: str, GCS path to KDE grayscale image.
        patch_coordinates: list, patch x and y coordinate 2-tuples.
        patch_height: int, the height in pixels of an image patch.
        patch_width: int, the width in pixels of an image patch.
        height_scale_factor: float, scale factor for height.
        width_scale_factor: float, scale factor for width.
        timeout: int, number of seconds to wait for an element to process.
        x_bounds: 2-tuple, the min and max bounds in the x-dimension of the
            original images.
        y_bounds: 2-tuple, the min and max bounds in the y-dimension of the
            original images.
    """
    def __init__(
        self,
        slide_name,
        annotations_image_gcs_path,
        kde_gs_image_gcs_path,
        patch_coordinates_gcs_path,
        patch_height,
        patch_width,
        height_scale_factor,
        width_scale_factor,
        timeout
    ):
        """Constructor of ParDo class that creates polygon confusion matrix.

        Args:
            slide_name: str, name of slide.
            annotations_image_gcs_path: str, GCS path to annotations image.
            kde_gs_image_gcs_path: str, GCS path to KDE grayscale image.
            patch_coordinates_gcs_path: str, GCS path to patch coordinates.
            patch_height: int, the height in pixels of an image patch.
            patch_width: int, the width in pixels of an image patch.
            height_scale_factor: float, scale factor for height.
            width_scale_factor: float, scale factor for width.
            timeout: int, number of seconds to wait for an element to process.
        """
        self.slide_name = slide_name
        self.annotations_image_gcs_path = annotations_image_gcs_path
        self.kde_gs_image_gcs_path = kde_gs_image_gcs_path
        self.patch_coordinates = self.get_patch_coords_from_gcs(
            gcs_path=patch_coordinates_gcs_path
        )
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.height_scale_factor = height_scale_factor
        self.width_scale_factor = width_scale_factor
        self.timeout = timeout

    def get_patch_coords_from_gcs(self, gcs_path):
        """Gets patch coordinates from GCS.

        Args:
            gcs_path: str, GCS path patch coordinate CSV is stored.

        Returns:
            List of 2-tuples of x and y patch corner coordinates.
        """
        from google.cloud import storage

        # Instantiate a Google Cloud Storage client.
        storage_client = storage.Client()
        # Specify required bucket and file.
        bucket = storage_client.get_bucket(gcs_path.split("/")[2])
        blob = bucket.blob("/".join(gcs_path.split("/")[3:]))

        # Download the contents of the blob as a string.
        data = blob.download_as_string(client=None)
        data_string = data.decode("utf-8")
        data_lines = data_string.split("\n")[:-1]
        data_list = []
        # Parse each line into a 2-tuple of int coordinates.
        return [
            (int(line.split(",")[0]), int(line.split(",")[1]))
            for line in data_lines
        ]

    def create_patch_polygon(self, patch_coords):
        """Creates polygon using patch corner vertices.

        Args:
            patch_coords: 2-tuple, x, y coordinates.

        Returns:
            `Polygon` with patch coordinates for exterior.
        """
        import shapely.geometry

        w, h = patch_coords
        w, h = w * self.width_scale_factor, h * self.height_scale_factor
        patch_width, patch_height = (
            self.patch_width * self.width_scale_factor,
            self.patch_height * self.height_scale_factor
        )
        top_left = shapely.geometry.Point(h, w)
        top_right = shapely.geometry.Point(h, w + patch_width)
        bottom_left = shapely.geometry.Point(h + patch_height, w)
        bottom_right = shapely.geometry.Point(
            h + patch_height, w + patch_width
        )
        return shapely.geometry.Polygon(
            shell=[top_left, top_right, bottom_right, bottom_left, top_left]
        )

    def create_patch_polygons(self):
        """Creates polygons everywhere there is a patch from list of coordinates.

        Returns:
            `MultiPolygon` encapsulating everywhere there is a patch.
        """
        import shapely.geometry

        return shapely.geometry.MultiPolygon(
            polygons=[
                self.create_patch_polygon(patch_coords)
                for patch_coords in self.patch_coordinates
            ]
        ).buffer(0.0)

    def process_image(self, image_gcs_path, channels):
        """Processes image into correct format.

        Args:
            image_gcs_path: str, GCS path to image.
            channels: int, number of channels of image.

        Returns:
            Processed image tensor in range [0., 1.].
        """
        image = tf.io.read_file(filename=image_gcs_path)
        return tf.cast(
            x=(255 - tf.io.decode_png(contents=image, channels=1)),
            dtype=tf.float32
        ).numpy() / 255.

    def create_original_multipolygon(self, image):
        """Creates original `MultiPolygon` of image.

        Args:
            image: tensor, image tensor in range [0., 1.].

        Returns:
            `MultiPolygon` object of image.
        """
        import shapely.affinity
        import shapely.geometry
        import shapely.ops
        import skimage.measure

        contours = skimage.measure.find_contours(
            image,
            level=0.5,
            fully_connected="low",
            positive_orientation="low",
            mask=None
        )
        multipolygon = shapely.geometry.MultiPolygon(
            shapely.ops.unary_union(
                shapely.geometry.MultiPolygon(
                    [
                        shapely.geometry.Polygon(contour)
                        for contour in contours
                    ]
                )
            )
        )
        if not isinstance(multipolygon, shapely.geometry.MultiPolygon):
            multipolygon = [multipolygon]
        return multipolygon

    def process_element(self, element, q):
        """Processes threshold and dilation factor to yield confusion metrics.

        Args:
            element: 2-tuple of threshold and dilation factor.
            q: `multiprocessing.Queue`, queue to hold processed elements.
        """
        import shapely.affinity
        import shapely.ops

        annotations_image = self.process_image(
            image_gcs_path=self.annotations_image_gcs_path, channels=1
        )[:, :, 0]
        kde_gs_image = self.process_image(
            image_gcs_path=self.kde_gs_image_gcs_path, channels=1
        )[:, :, 0]

        print("Making annotations multipolygon")
        annotations_multipolygon = self.create_original_multipolygon(
            image=annotations_image
        ).buffer(0.0)

        threshold, dilation_factor = element

        print("Making kde_gs_thresholded multipolygon")
        kde_gs_thresholded_multipolygon = self.create_original_multipolygon(
            image=kde_gs_image > threshold
        )

        print("Making prediction multipolygon")
        prediction_multipolygon = shapely.ops.unary_union(
            [
                shapely.affinity.scale(
                    geom=polygon, xfact=dilation_factor, yfact=dilation_factor
                )
                for polygon in kde_gs_thresholded_multipolygon
            ]
        ).buffer(0.0)

        print("Making patch multipolygon")
        patch_multipolygon = self.create_patch_polygons().buffer(0.0)

        print("Bounding prediction multipolygon")
        prediction_multipolygon = prediction_multipolygon.intersection(
            other=patch_multipolygon
        )

        print("Calculating confusion metrics")
        num_polygons = (
            1 if not isinstance(
                prediction_multipolygon, shapely.geometry.MultiPolygon
            )
            else len(prediction_multipolygon)
        )
        true_positives = prediction_multipolygon.intersection(
            other=annotations_multipolygon
        )
        false_positives = prediction_multipolygon.difference(
            other=true_positives
        )
        false_negatives = annotations_multipolygon.difference(
            other=true_positives
        )
        not_true_negatives_area = (
            true_positives.area + false_positives.area + false_negatives.area
        )
        true_negatives_area = patch_multipolygon.area - not_true_negatives_area

        confusion_matrix_dict = {
            "slide_name": self.slide_name,
            "threshold": threshold,
            "dilation_factor": dilation_factor,
            "num_polygons": num_polygons,
            "true_positives": true_positives.area,
            "false_positives": false_positives.area,
            "false_negatives": false_negatives.area,
            "true_negatives": true_negatives_area
        }

        if q is None:
            return confusion_matrix_dict
        print("Adding to multiprocessing queue")
        q.put(confusion_matrix_dict)

    def process(self, element):
        """Processes threshold and dilation factor to yield confusion metrics.

        Args:
            element: 2-tuple of threshold and dilation factor.

        Returns:
            Dictionary of confusion metrics at element's threshold and
                dilation factor.
        """
        import logging
        import multiprocessing

        if self.timeout > 0:
            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=self.process_element,
                args=(element, q)
            )
            p.start()

            p.join(self.timeout)

            if p.is_alive():
                logging.info("Timeout reached! Skipping polygon!")
                p.terminate()
                p.join()
                yield {}
            else:
                yield q.get()
        else:
            yield self.process_element(element, None)


class LocalPolygonDoFn(PolygonDoFn):
    """Local class that creates KDE & annotation polygon confusion matrix.

    Attributes:
        kde_gs_image: tensor, KDE grayscale image tensor of shape
            (height, width).
        annotations_polygon: `MultiPolygon`, polygons of annotation image.
        patch_multipolygon: `MultiPolygon`, polygons of patch coordinates.
        threshold: float, threshold to apply to KDE grayscale image.
        kde_gs_thresholded_multipolygon: `MultiPolygon`, polygons of
            thresholded KDE grayscale image.
    """
    def __init__(
        self,
        slide_name,
        annotations_image_gcs_path,
        kde_gs_image_gcs_path,
        patch_coordinates_gcs_path,
        patch_height,
        patch_width,
        height_scale_factor,
        width_scale_factor,
        timeout
    ):
        """Constructor of ParDo class that creates polygon confusion matrix.

        Args:
            slide_name: str, name of slide.
            annotations_image_gcs_path: str, GCS path to annotations image.
            kde_gs_image_gcs_path: str, GCS path to KDE grayscale image.
            patch_coordinates_gcs_path: str, GCS path to patch coordinates.
            patch_height: int, the height in pixels of an image patch.
            patch_width: int, the width in pixels of an image patch.
            height_scale_factor: float, scale factor for height.
            width_scale_factor: float, scale factor for width.
            timeout: int, number of seconds to wait for an element to process.
        """
        super().__init__(
            slide_name,
            annotations_image_gcs_path,
            kde_gs_image_gcs_path,
            patch_coordinates_gcs_path,
            patch_height,
            patch_width,
            height_scale_factor,
            width_scale_factor,
            timeout
        )
        annotations_image = self.process_image(
            image_gcs_path=self.annotations_image_gcs_path, channels=1
        )[:, :, 0]
        self.kde_gs_image = self.process_image(
            image_gcs_path=self.kde_gs_image_gcs_path, channels=1
        )[:, :, 0]

        print("Making annotations multipolygon")
        self.annotations_multipolygon = self.create_original_multipolygon(
            image=annotations_image
        ).buffer(0.0)

        print("Making patch multipolygon")
        self.patch_multipolygon = self.create_patch_polygons().buffer(0.0)
        self.threshold = 0.0
        self.kde_gs_thresholded_multipolygon = None

    def apply_threshold(self, threshold):
        """Applies threshold to KDE grayscale image.

        Args:
            threshold: float, threshold to apply to KDE grayscale image.
        """
        self.threshold = threshold
        print("Making kde_gs_thresholded multipolygon")
        self.kde_gs_thresholded_multipolygon = (
            self.create_original_multipolygon(
                image=self.kde_gs_image > threshold
            )
        )

    def get_polygon_confusion_matrix_dict(self, dilation_factor):
        """Gets polygon confusion matrix dictionary.

        Args:
            dilation_factor: float, how much to scale/dilate each prediction
                polygon.

        Returns:
            Dictionary of confusion matrix metrics.
        """
        import shapely.affinity
        import shapely.ops

        print("Making prediction multipolygon")
        prediction_multipolygon = shapely.ops.unary_union(
            [
                shapely.affinity.scale(
                    geom=polygon, xfact=dilation_factor, yfact=dilation_factor
                )
                for polygon in self.kde_gs_thresholded_multipolygon
            ]
        ).buffer(0.0)

        print("Bounding prediction multipolygon")
        prediction_multipolygon = prediction_multipolygon.intersection(
            other=self.patch_multipolygon
        )

        print("Calculating confusion metrics")
        num_polygons = (
            1 if not isinstance(
                prediction_multipolygon, shapely.geometry.MultiPolygon
            )
            else len(prediction_multipolygon)
        )
        true_positives = prediction_multipolygon.intersection(
            other=self.annotations_multipolygon
        )
        false_positives = prediction_multipolygon.difference(
            other=true_positives
        )
        false_negatives = self.annotations_multipolygon.difference(
            other=true_positives
        )
        not_true_negatives_area = (
            true_positives.area + false_positives.area + false_negatives.area
        )
        true_negatives_area = (
            self.patch_multipolygon.area - not_true_negatives_area
        )

        return {
            "slide_name": self.slide_name,
            "threshold": self.threshold,
            "dilation_factor": dilation_factor,
            "num_polygons": num_polygons,
            "true_positives": true_positives.area,
            "false_positives": false_positives.area,
            "false_negatives": false_negatives.area,
            "true_negatives": true_negatives_area
        }
