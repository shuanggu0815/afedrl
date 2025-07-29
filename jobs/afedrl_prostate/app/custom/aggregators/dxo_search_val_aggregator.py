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
from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
from typing import Any, Dict, Optional

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.app_common.app_constant import AppConstants

class DXOSearchValAggregator(DXOAggregator):
    def __init__(
        self,
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[Dict[str, Any]] = None,
        expected_data_kind: DataKind = DataKind.WEIGHT_DIFF,
        name_postfix: str = "",
        weigh_by_local_iter: bool = True,
    ):
        super().__init__()
        self.expected_data_kind = expected_data_kind

    def accept(self, dxo: DXO, contributor_name, contribution_round, fl_ctx: FLContext) -> bool:
        """Store DXO and update aggregator's internal state
        Args:
            dxo: information from contributor
            contributor_name: name of the contributor
            contribution_round: round of the contribution
            fl_ctx: context provided by workflow
        Returns:
            The boolean to indicate if DXO is accepted.
        """
        self.log_info(fl_ctx, "self.expected_data_kind got {}".format(self.expected_data_kind))
        if not isinstance(dxo, DXO):
            self.log_error(fl_ctx, f"Expected DXO but got {type(dxo)}")
            return False

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.METRICS):
            self.log_error(fl_ctx, "cannot handle data kind {}".format(dxo.data_kind))
            return False

        if dxo.data_kind != self.expected_data_kind:
            self.log_error(fl_ctx, "expected {} but got {}".format(self.expected_data_kind, dxo.data_kind))
            return False

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            if self.processed_algorithm is None:
                self.processed_algorithm = processed_algorithm
            elif self.processed_algorithm != processed_algorithm:
                self.log_error(
                    fl_ctx,
                    f"Only supports aggregation of data processed with the same algorithm ({self.processed_algorithm}) "
                    f"but got algorithm: {processed_algorithm}",
                )
                return False

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        data = dxo.data
        if data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False

        for item in self.aggregation_helper.get_history():
            if contributor_name == item["contributor_name"]:
                prev_round = item["round"]
                self.log_warning(
                    fl_ctx,
                    f"discarding DXO from {contributor_name} at round: "
                    f"{contribution_round} as {prev_round} accepted already",
                )
                return False

        n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
        if n_iter is None:
            if self.warning_count.get(contributor_name, 0) <= self.warning_limit:
                self.log_warning(
                    fl_ctx,
                    f"NUM_STEPS_CURRENT_ROUND missing in meta of DXO"
                    f" from {contributor_name} and set to default value, 1.0. "
                    f" This kind of message will show {self.warning_limit} times at most.",
                )
                if contributor_name in self.warning_count:
                    self.warning_count[contributor_name] = self.warning_count[contributor_name] + 1
                else:
                    self.warning_count[contributor_name] = 0
            n_iter = 1.0
        float_n_iter = float(n_iter)
        aggregation_weight = self.aggregation_weights.get(contributor_name)
        if aggregation_weight is None:
            if self.warning_count.get(contributor_name, 0) <= self.warning_limit:
                self.log_warning(
                    fl_ctx,
                    f"Aggregation_weight missing for {contributor_name} and set to default value, 1.0"
                    f" This kind of message will show {self.warning_limit} times at most.",
                )
                if contributor_name in self.warning_count:
                    self.warning_count[contributor_name] = self.warning_count[contributor_name] + 1
                else:
                    self.warning_count[contributor_name] = 0
            aggregation_weight = 1.0

        # aggregate
        self.aggregation_helper.add(data, aggregation_weight * float_n_iter, contributor_name, contribution_round)
        # self.log_info(fl_ctx,f"contributor_name: {self.aggregation_helper.history}")
        self.log_debug(fl_ctx, "End accept")
        return True