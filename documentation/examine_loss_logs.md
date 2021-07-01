# Proganomaly Loss Logs Examinations

## License

Copyright 2020 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at [Apache License Page](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

## Parameters 
- There is only one parameter you need to provide, which is the path to your exported loss logs folder output from 
your trained model.
######
        %%bash
        rm -rf loss_logs/*
        gsutil -m cp -r gs://.../trained_models/loss_logs . >/dev/null 2>&1