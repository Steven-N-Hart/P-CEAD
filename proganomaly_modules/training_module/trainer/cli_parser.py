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

import argparse
import json


def parse_arguments(parser):
    """Parses command line arguments.

    Args:
        parser: instance of `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--job-dir",
        help="This model ignores this field, but it is required by gcloud.",
        type=str,
        default="junk"
    )
    parser.add_argument(
        "--json_config_gcs_path",
        help="GCS path to JSON config file.",
        type=str,
        default=""
    )
    parser.add_argument(
        "--json_overrides",
        help="Serialized nested dictionary of overrides of JSON config.",
        type=str,
        default=""
    )

def parse_command_line_arguments():
    """Parses command line arguments and returns dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add arguments to parser.
    parse_arguments(parser)

    # Parse all arguments.
    args = parser.parse_args()
    arguments = args.__dict__

    return arguments
