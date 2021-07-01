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

import tensorflow as tf


def preprocess_images(images):
    """Preprocesses images array.

    Args:
        images: np.array or tensor, images of shape
            (batch_size, height width, depth).

    Returns:
        Tensor of images of shape (batch_size, height width, depth).
    """
    # Convert image values from [0, 255] to [-1., 1.].
    images = (tf.cast(x=images, dtype=tf.float32) - 127.5) / 127.5

    return images

def in_core_dataset(
    images, labels, dataset_len, batch_size, block_idx, params, training
):
    """Gets in-core dataset.

    Args:
        images: tensor, images of shape
            (batch_size, dataset_height, dataset_width, depth).
        labels: tensor, int labels of shape (batch_size,).
        dataset_len: int, number of examples in dataset.
        batch_size: int, number of examples per batch.
        block_idx: int, current resolution block index.
        params: dict, user passed parameters.
        training: bool, if training or not.

    Returns:
        Dataset of dictionary of images tensor of shape
            (batch_size, height, width, depth) and labels tensor of shape
            (batch_size,).
    """
    def resize_images(image, label):
        """Resizes images tensor.

        Args:
            image: tensor, images of shape
                (batch_size, dataset_height, dataset_width, depth).
            label: tensor, int labels of shape (batch_size,).

        Returns:
            Dictionary of images tensor of shape
                (batch_size, size, size, depth) and labels tensor of shape
                (batch_size,).
        """
        size = 4 * 2 ** block_idx
        image = tf.image.resize(
                images=image, size=(size, size), method="nearest"
            )

        return {"image": image}, label

    dataset = tf.data.Dataset.from_tensor_slices(tensors=(images, labels))

    if training:
        dataset = dataset.repeat(count=None).shuffle(buffer_size=dataset_len)

    dataset = dataset.batch(batch_size=batch_size)

    # Resize images for resolution block.
    dataset = dataset.map(
        map_func=resize_images,
        num_parallel_calls=(
            tf.data.experimental.AUTOTUNE
            if params["input_fn_autotune"]
            else None
        ),
        deterministic=False
    )

    # Prefetch data to improve latency.
    dataset = dataset.prefetch(
        buffer_size=(
            tf.data.experimental.AUTOTUNE
            if params["input_fn_autotune"]
            else 1
        )
    )

    return dataset

def mnist_dataset(batch_size, block_idx, params, training):
    """Gets tf.data.Dataset using in-core MNIST dataset.

    Args:
        batch_size: int, number of examples per batch.
        block_idx: int, current resolution block index.
        params: dict, user passed parameters.
        training: bool, if training or not.

    Returns:
        An input function.
    """
    def preprocess_mnist_images(images):
        """Preprocesses specifically for MNIST images.

        Args:
            images: np.array, array of images of shape (60000, 28, 28).

        Returns:
            Tensor of images of shape (60000, 32, 32, 1).
        """
        # Pad 28x28 images to 32x32.
        images = tf.pad(
            tensor=images,
            paddings=[[0, 0], [2, 2], [2, 2]],
            mode="CONSTANT",
            constant_values=0
        )

        # Add dimension for num_channels(1) to end.
        images = tf.expand_dims(input=images, axis=-1)

        return images

    def _input_fn():
        """Wrapper input function to get data tensors.

        Returns:
            Batched dataset object of dictionary of image tensors and label
                tensor.
        """
        if training:
            (train_images, train_labels), (_, _) = (
                tf.keras.datasets.mnist.load_data()
            )
            images = preprocess_images(
                images=preprocess_mnist_images(images=train_images)
            )
            labels = train_labels
        else:
            (_, _), (test_images, test_labels) = (
                tf.keras.datasets.mnist.load_data()
            )
            images = preprocess_images(
                images=preprocess_mnist_images(images=test_images)
            )
            labels = test_labels
        print("MNIST dataset shape = {}".format(images.shape))

        return in_core_dataset(
            images=images,
            labels=labels,
            dataset_len=50000,
            batch_size=batch_size,
            block_idx=block_idx,
            params=params,
            training=training
        )

    return _input_fn

def cifar10_dataset(batch_size, block_idx, params, training):
    """Gets tf.data.Dataset using in-core CIFAR-10 dataset.

    Args:
        batch_size: int, number of examples per batch.
        block_idx: int, current resolution block index.
        params: dict, user passed parameters.
        training: bool, if training or not.

    Returns:
        An input function.
    """
    def filter_cifar_images_by_class(images, labels, class_idx):
        """Filters CIFAR-10 images to only chosen class.

        Args:
            images: np.array, array of images of shape (50000, 32, 32, 3).
            labels: np.array, array of int labels of shape (50000,).
            class_idx: int, index of chosen class.

        Returns:
            Numpy array of chosen class images of shape (5000, 32, 32, 3).
        """
        in_class_images = images[labels.flatten() == class_idx, :, :, :]

        return in_class_images

    def preprocess_cifar10_images(images, labels):
        """Preprocesses specifically for CIFAR-10 images.

        Args:
            images: np.array, array of images of shape (50000, 32, 32, 3).
            labels: np.array, array of int labels of shape (50000,).

        Returns:
            images: tensor, images either of shape (5000, 32, 32, 3) with
                filtering to one class or (50000, 32, 32, 3) if otherwise.
            labels: tensor, labels either of shape (5000,) with filtering to
                one class or (50000,) if otherwise.
        """
        if params["dataset"] == "cifar10_car":
            images = filter_cifar_images_by_class(
                images=images, labels=labels, class_idx=1
            )

            labels = tf.tile(input=[1], multiples=[images.shape[0]])

        return images, labels

    def _input_fn():
        """Wrapper input function to get data tensors.

        Returns:
            Batched dataset object of dictionary of image tensors and label
                tensor.
        """
        if training:
            (train_images, train_labels), (_, _) = (
                tf.keras.datasets.cifar10.load_data()
            )
            images, labels = preprocess_cifar10_images(
                images=train_images, labels=train_labels
            )
        else:
            (_, _), (test_images, test_labels) = (
                tf.keras.datasets.cifar10.load_data()
            )
            images, labels = preprocess_cifar10_images(
                images=test_images, labels=test_labels
            )
        images = preprocess_images(images=images)
        print("CIFAR-10 dataset shape = {}".format(images.shape))

        return in_core_dataset(
            images=images,
            labels=labels,
            dataset_len=50000,
            batch_size=batch_size,
            block_idx=block_idx,
            params=params,
            training=training
        )

    return _input_fn

def read_tf_record_dataset(
    file_pattern, batch_size, block_idx, params, training
):
    """Reads TF Record data using tf.data, doing necessary preprocessing.

    Given filename, mode, batch size, and other parameters, read TF Record
    dataset using Dataset API, apply necessary preprocessing, and return an
    input function to the Estimator API.

    Args:
        file_pattern: str, file pattern that to read into our tf.data dataset.
        batch_size: int, number of examples per batch.
        block_idx: int, the current resolution block index.
        params: dict, dictionary of user passed parameters.
        training: bool, if training or not.

    Returns:
        An input function.
    """
    def fetch_dataset(filename):
        """Fetches TFRecord Dataset from given filename.

        Args:
            filename: str, name of TFRecord file.

        Returns:
            Dataset containing TFRecord Examples.
        """
        buffer_size = 8 * 1024 * 1024  # 8 MiB per file
        dataset = tf.data.TFRecordDataset(
            filenames=filename,
            buffer_size=(
                None
                if params["input_fn_autotune"]
                else buffer_size
            ),
            num_parallel_reads=8
        )

        return dataset

    def decode_example(protos, block_idx, params):
        """Decodes TFRecord file into tensors.

        Given protobufs, decode into image and label tensors.

        Args:
            protos: protobufs from TFRecord file.
            block_idx: int, the current resolution block index.
            params: dict, user passed parameters.

        Returns:
            Image and label tensors.
        """
        dtype_map = {
            "str": tf.string,
            "int": tf.int64,
            "float": tf.float32
        }

        # Create feature schema map for protos.
        tf_example_features = {
            feat["name"]: (
                tf.io.FixedLenFeature(
                    shape=feat["shape"], dtype=dtype_map[feat["dtype"]]
                )
                if feat["type"] == "FixedLen"
                else tf.io.FixedLenSequenceFeature(
                    shape=feat["shape"], dtype=dtype_map[feat["dtype"]]
                )
            )
            for feat in params["tf_record_example_schema"]
        }

        # Parse features from tf.Example.
        parsed_features = tf.io.parse_single_example(
            serialized=protos, features=tf_example_features
        )

        # Convert from a scalar string tensor (whose single string has
        # length height * width * depth) to a uint8 tensor with shape
        # (height * width * depth).
        if params["image_encoding"] == "raw":
            image = tf.io.decode_raw(
                input_bytes=parsed_features[params["image_feature_name"]],
                out_type=tf.uint8
            )
        elif params["image_encoding"] == "png":
            image = tf.io.decode_png(
                contents=parsed_features[params["image_feature_name"]],
                channels=params["image_depth"]
            )
        elif params["image_encoding"] == "jpeg":
            image = tf.io.decode_jpeg(
                contents=parsed_features[params["image_feature_name"]],
                channels=params["image_depth"]
            )

        # Reshape flattened image back into normal dimensions.
        if params["use_multiple_resolution_records"]:
            height, width = params["projection_dims"][0:2]
            height *= (2 ** block_idx)
            width *= (2 ** block_idx)
            image = tf.reshape(
                tensor=image, shape=(height, width, params["image_depth"])
            )
        else:
            image = tf.reshape(
                tensor=image,
                shape=(
                    params["image_predownscaled_height"],
                    params["image_predownscaled_width"],
                    params["image_depth"]
                )
            )

        # Preprocess image.
        image = preprocess_images(images=image)

        if params["label_feature_name"]:
            # Convert label from a scalar uint8 tensor to an int32 scalar.
            label = tf.cast(x=parsed_features["label"], dtype=tf.int32)

            return {"image": image}, label
        return {"image": image}

    def set_static_shape(features, labels, batch_size, params):
        """Sets static shape of batched input tensors in dataset.

        Args:
            features: dict, keys are feature names and values are tensors.
            labels: tensor, label data.
            batch_size: int, number of examples per batch.
            params: dict, user passed parameters.

        Returns:
            Features tensor dictionary and labels tensor.
        """
        features["image"].set_shape(
            features["image"].get_shape().merge_with(
                tf.TensorShape(dims=(batch_size, None, None, None))
            )
        )

        if params["label_feature_name"]:
            labels.set_shape(
                labels.get_shape().merge_with(
                    tf.TensorShape(dims=(batch_size))
                )
            )

            return features, labels
        return features

    def _input_fn():
        """Wrapper input function used by Estimator API to get data tensors.

        Returns:
            Batched dataset object of dictionary of feature tensors and label
                tensor.
        """
        # Create dataset to contain list of files matching pattern.
        dataset = tf.data.Dataset.list_files(
            file_pattern=file_pattern, shuffle=training
        )

        # Repeat dataset files indefinitely if in training.
        if training:
            dataset = dataset.repeat()

        # Parallel interleave multiple files at once with map function.
        dataset = dataset.interleave(
            map_func=fetch_dataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False
        )

        # Shuffle the Dataset TFRecord Examples if in training.
        if training:
            dataset = dataset.shuffle(buffer_size=1024)

        # Decode TF Record Example into a features dictionary of tensors.
        dataset = dataset.map(
            map_func=lambda x: decode_example(
                protos=x,
                block_idx=block_idx,
                params=params
            ),
            num_parallel_calls=(
                tf.data.experimental.AUTOTUNE
                if params["input_fn_autotune"]
                else None
            )
        )

        # Batch dataset and drop remainder so there are no partial batches.
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        # Assign static shape, namely make the batch size axis static.
        if params["label_feature_name"]:
            dataset = dataset.map(
                map_func=lambda x, y: set_static_shape(
                    features=x,
                    labels=y,
                    batch_size=batch_size,
                    params=params
                ),
                num_parallel_calls=(
                    tf.data.experimental.AUTOTUNE
                    if params["input_fn_autotune"]
                    else None
                )
            )
        else:
            dataset = dataset.map(
                map_func=lambda x: set_static_shape(
                    features=x,
                    labels=None,
                    batch_size=batch_size,
                    params=params
                ),
                num_parallel_calls=(
                    tf.data.experimental.AUTOTUNE
                    if params["input_fn_autotune"]
                    else None
                )
            )

        # Prefetch data to improve latency.
        dataset = dataset.prefetch(
            buffer_size=(
                tf.data.experimental.AUTOTUNE
                if params["input_fn_autotune"]
                else 1
            )
        )

        return dataset

    return _input_fn
