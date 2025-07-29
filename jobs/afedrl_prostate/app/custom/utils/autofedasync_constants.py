# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


class AutoFedRLConstants(object):
    HYPERPARAMTER_COLLECTION = "hyperparamter_collection"
    GLOBAL_HYPERPARAMTER_COLLECTION = "global_hyperparamter_collection"
    CLIENT_HYPERPARAMTER_COLLECTION = "client_hyperparamter_collection"
    MODEL_VALIDATE_FOR_SEARCH = "model_validate_for_hp_search"
    SEARCH_SPACE_ID = "search_space_id"
    VAL_ROUND = "val_round"
    TRAIN_AND_VALIDATE = "train_and_validate"
