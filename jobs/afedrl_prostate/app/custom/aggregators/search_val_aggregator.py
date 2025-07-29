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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.apis.dxo import DXO, DataKind, from_shareable

from typing import Any, Dict, Union

from aggregators.dxo_search_val_aggregator import DXOSearchValAggregator

def _is_nested_aggregation_weights(aggregation_weights):
    if not aggregation_weights:
        return False
    if not isinstance(aggregation_weights, dict):
        return False
    first_value = next(iter(aggregation_weights.items()))[1]
    if not isinstance(first_value, dict):
        return False
    return True


def _get_missing_keys(ref_dict: dict, dict_to_check: dict):
    result = []
    for k in ref_dict:
        if k not in dict_to_check:
            result.append(k)
    return result

class SearchValAggregator(InTimeAccumulateWeightedAggregator):
    def __init__(
        self,
        exclude_vars: Union[str, Dict[str, str], None] = None,
        aggregation_weights: Union[Dict[str, Any], Dict[str, Dict[str, Any]], None] = None,
        expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHT_DIFF,
        weigh_by_local_iter: bool = True,
    ):
        super().__init__()
        self.logger.debug(f"exclude vars: {exclude_vars}")
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")
        self.logger.debug(f"expected data kind: {expected_data_kind}")

        self._single_dxo_key = ""
        self._weigh_by_local_iter = weigh_by_local_iter

        self.aggregation_weights = aggregation_weights
        self.exclude_vars = exclude_vars
        self.expected_data_kind = expected_data_kind
    """Perform accumulated weighted aggregation with support for updating aggregation weights.

    Shares arguments with base class
    """

    def _initialize(self, aggregation_weights, exclude_vars, expected_data_kind):
        # Check expected data kind
        if isinstance(expected_data_kind, dict):
            for k, v in expected_data_kind.items():
                if v not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.METRICS]:
                    raise ValueError(
                        f"expected_data_kind[{k}] = {v} is not {DataKind.WEIGHT_DIFF} or {DataKind.WEIGHTS} or {DataKind.METRICS}"
                    )
            self.expected_data_kind = expected_data_kind
        else:
            if expected_data_kind not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.METRICS]:
                raise ValueError(
                    f"expected_data_kind = {expected_data_kind} is not {DataKind.WEIGHT_DIFF} or {DataKind.WEIGHTS} or {DataKind.METRICS}"
                )
            self.expected_data_kind = {self._single_dxo_key: expected_data_kind}
        # Check exclude_vars
        if exclude_vars:
            if not isinstance(exclude_vars, dict) and not isinstance(exclude_vars, str):
                raise ValueError(
                    f"exclude_vars = {exclude_vars} should be a regex string but got {type(exclude_vars)}."
                )
            if isinstance(exclude_vars, dict):
                missing_keys = _get_missing_keys(expected_data_kind, exclude_vars)
                if len(missing_keys) != 0:
                    raise ValueError(
                        "A dict exclude_vars should specify exclude_vars for every key in expected_data_kind. "
                        f"But missed these keys: {missing_keys}"
                    )
        exclude_vars_dict = dict()
        for k in self.expected_data_kind.keys():
            if isinstance(exclude_vars, dict):
                if k in exclude_vars:
                    if not isinstance(exclude_vars[k], str):
                        raise ValueError(
                            f"exclude_vars[{k}] = {exclude_vars[k]} should be a regex string but got {type(exclude_vars[k])}."
                        )
                    exclude_vars_dict[k] = exclude_vars[k]
            else:
                # assume same exclude vars for each entry of DXO collection.
                exclude_vars_dict[k] = exclude_vars
        if self._single_dxo_key in self.expected_data_kind:
            exclude_vars_dict[self._single_dxo_key] = exclude_vars
        self.exclude_vars = exclude_vars_dict
        # Check aggregation weights
        if _is_nested_aggregation_weights(aggregation_weights):
            missing_keys = _get_missing_keys(expected_data_kind, aggregation_weights)
            if len(missing_keys) != 0:
                raise ValueError(
                    "A dict of dict aggregation_weights should specify aggregation_weights "
                    f"for every key in expected_data_kind. But missed these keys: {missing_keys}"
                )
        aggregation_weights = aggregation_weights or {}
        aggregation_weights_dict = dict()
        for k in self.expected_data_kind.keys():
            if k in aggregation_weights:
                aggregation_weights_dict[k] = aggregation_weights[k]
            else:
                # assume same aggregation weights for each entry of DXO collection.
                aggregation_weights_dict[k] = aggregation_weights
        self.aggregation_weights = aggregation_weights_dict
        print(f"self.expected_data_kind: {self.expected_data_kind}")
        # Set up DXO aggregators
        self.dxo_aggregators = dict()
        for k in self.expected_data_kind.keys():
            self.dxo_aggregators.update(
                {
                    k: DXOSearchValAggregator(
                        exclude_vars=self.exclude_vars[k],
                        aggregation_weights=self.aggregation_weights[k],
                        expected_data_kind=self.expected_data_kind[k],
                        name_postfix=k,
                        weigh_by_local_iter=self._weigh_by_local_iter,
                    )
                }
            )

    def get_len(self):
        for key in self.expected_data_kind.keys():
            length = self.dxo_aggregators[key].aggregation_helper.get_len()
        return length