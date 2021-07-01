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

import json
import os
import tensorflow as tf

from . import cli_parser
from . import defaults
from . import model


def override_config(config, overrides):
    """Overrides config dictionary.

    Args:
        config: dict, user passed parameters for training job from JSON or
            defaults.
        overrides: dict, user passed parameters to override some within config
            dictionary.

    Returns:
        Modified in-place config dictionary by overrides.
    """
    for k, v in overrides.items():
        if isinstance(v, dict):
            config[k] = override_config(config.get(k, {}), v)
        else:
            config[k] = v

    return config

def get_config(arguments):
    """Gets config from parsed arguments.

    Args:
        arguments: dict, parsed arguments from the command-line.

    Returns:
        Dictionary containing user passed parameters for the training job.
    """
    if arguments["json_config_gcs_path"]:
        filename = arguments["json_config_gcs_path"]
        with tf.io.gfile.GFile(name=filename, mode="r") as f:
            config = json.load(f)
    else:
        config = defaults.get_default_config()

    if arguments["json_overrides"]:
        overrides = (
            arguments["json_overrides"].replace(";", " ").replace("\'", "\"")
        )
        print("task.py: overrides = {}".format(overrides))
        config = override_config(
            config=config, overrides=json.loads(overrides)
        )

    return config

if __name__ == "__main__":
    # Parse command line arguments.
    arguments = cli_parser.parse_command_line_arguments()

    # Unused args provided by service.
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)
    print("task.py: arguments = {}".format(arguments))

    # Get config for training job from parsed arguments.
    config = get_config(arguments)

    # Append trial_id to path if we are doing hptuning.
    # This code can be removed if you are not using hyperparameter tuning.
    config["training"]["output_dir"] = os.path.join(
        config["training"]["output_dir"],
        json.loads(
            os.environ.get(
                "TF_CONFIG", "{}"
            )
        ).get("task", {}).get("trial", ""))

    print("task.py: config = {}".format(config))

    # Instantiate instance of model train and evaluate loop.
    train_and_evaluate_model = model.TrainAndEvaluateModel(params=config)

    # Run the training job.
    train_and_evaluate_model.train_and_evaluate()
