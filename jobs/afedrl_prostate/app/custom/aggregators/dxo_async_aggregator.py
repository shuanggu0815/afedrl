# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import torch
import os
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_constant import FLContextKey
from aggregators.async_aggregation_helper import AsyncAggregationHelper
from nvflare.app_common.app_constant import AppConstants


class DXOAsyncAggregator(FLComponent):
    def __init__(
        self,
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[Dict[str, Any]] = None,
        expected_data_kind: DataKind = DataKind.WEIGHT_DIFF,
        name_postfix: str = "",
        weigh_by_local_iter: bool = True,
    ):
        """Perform weighted aggregation for one kind of corresponding DXO from contributors.

        Args:
            exclude_vars (str, optional): Regex to match excluded vars during aggregation. Defaults to None.
            aggregation_weights (Dict[str, Any], optional): Aggregation weight for each contributor.
                                Defaults to None.
            expected_data_kind (DataKind): Expected DataKind for this DXO.
            name_postfix: optional postfix to give to class name and show in logger output.
            weigh_by_local_iter (bool, optional): Whether to weight the contributions by the number of iterations
                performed in local training in the current round. Defaults to `False`.
        """
        super().__init__()
        self.is_initialized = False
        self.expected_data_kind = expected_data_kind
        self.aggregation_weights = aggregation_weights or {}
        self.aggregation_weights_sm = self.aggregation_weights
        # self.num_data = {
        #     "site-1": 0.195153061,
        #     "site-2": 0.234693878,
        #     "site-3": 0.142857143,
        #     "site-4": 0.239795918,
        #     "site-5": 0.105867347,
        #     "site-6": 0.081632653
        # }
        self.num_data = {
            "client_I2CVB": 0.5,
            "client_MSD": 0.5,
            "client_NCI_ISBI_Dx": 0.5,
            "client_Promise12": 0.5,
            "client_PROSTATEx": 0.5,
            "client_NCI_ISBI_3T": 0.5

        }
        self.aggregation_weights = self.num_data.copy()
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")

        self.aggregation_helper = AsyncAggregationHelper(
            exclude_vars=exclude_vars, weigh_by_local_iter=weigh_by_local_iter
        )

        self.warning_count = {}
        self.warning_limit = 10
        self.processed_algorithm = None
        self.root = None
        self.writer = None
        if name_postfix:
            self._name += name_postfix
            self.logger = logging.getLogger(self._name)

    def initialize(self, fl_ctx: FLContext):
        self.root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.writer = SummaryWriter(self.root)

    def reset_aggregation_helper(self):
        if self.aggregation_helper:
            self.aggregation_helper.reset_stats()

    def accept(self, dxo: DXO, contributor_name, contribution_round, fl_ctx: FLContext, site_round: Optional[dict]={}) -> bool:
        """Store DXO and update aggregator's internal state
        Args:
            dxo: information from contributor
            contributor_name: name of the contributor
            contribution_round: round of the contribution
            fl_ctx: context provided by workflow
            site_round
        Returns:
            The boolean to indicate if DXO is accepted.
        """
        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)
            
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
        self.log_info(fl_ctx,f"before aggregation_weights:{self.aggregation_weights}")
        new_weight = data["aw"]
        self.log_info(fl_ctx,f"contributor:{contributor_name}, weight:{new_weight}")
        self.aggregation_weights[contributor_name] = new_weight
        self.log_info(fl_ctx,f"get weight aggregation_weights:{self.aggregation_weights}")
        self.aggregation_weights_sm = self.aggregation_weights.copy()
        aws = list(self.aggregation_weights_sm.values())
        aws_tensor = torch.tensor(aws, dtype=torch.float32)
        # scaled_aws_tensor = aws_tensor * 2
        scaled_aws_tensor = aws_tensor
        # softmax_weights = aws_tensor
        softmax_weights = F.softmax(scaled_aws_tensor, dim=0) 
        self.aggregation_weights_sm.update(
            {site: softmax_weights[idx].item() for idx, site in enumerate(self.aggregation_weights_sm.keys())}
        )
        self.log_info(fl_ctx,f"aggregation_weights:{self.aggregation_weights}")
        self.log_info(fl_ctx,f"aggregation_weights_sm:{self.aggregation_weights_sm}")

        self.writer.add_scalars("AggregationWeights", self.aggregation_weights_sm, current_round)

        aggregation_weight = self.aggregation_weights_sm.get(contributor_name)
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
        self.log_info(fl_ctx,f"weight each iter: {aggregation_weight * float_n_iter}")
        self.log_info(fl_ctx,f"contribution_round: {contribution_round}, site_round: {site_round}")
        self.aggregation_helper.add(data, aggregation_weight * float_n_iter, contributor_name, contribution_round, site_round)
        self.log_debug(fl_ctx, "End accept")
        return True

    
    def get_len(self):
        return self.aggregation_helper.get_len()

    def aggregate(self, fl_ctx: FLContext, global_weight: Shareable) -> Tuple[str, DXO]:
        """Called when workflow determines to generate DXO to send back to contributors
        Args:
            fl_ctx (FLContext): context provided by workflow
        Returns:
            str: The name of the client that has completed its task.
            DXO: the weighted mean of accepted DXOs from contributors
        """
        dox_global_weight = from_shareable(global_weight)
        dict_global_weight = dox_global_weight.data
        self.log_debug(fl_ctx, f"Start aggregation with weights {self.aggregation_weights}")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_debug(fl_ctx, f"complete history {self.aggregation_helper.get_len()}")
        agg_client, aggregated_dict = self.aggregation_helper.get_result(fl_ctx, dict_global_weight,current_round, self.aggregation_weights)
        if current_round <= 1800 and current_round > 1200:
            tensor_dict = {}
            tensor_dict = {key: torch.tensor(value) for key, value in aggregated_dict.items()}
            file_name = f"global_model_r_{current_round}.pt"
            save_path = os.path.join(self.root, file_name)
            torch.save(tensor_dict, save_path)

        contributor_name = agg_client["contributor_name"]
        site_round = agg_client["site_round"]
        self.log_info(fl_ctx, f"aggregate {contributor_name}'s(site_round:{site_round}) weights at round {current_round}")
        self.log_debug(fl_ctx, "End aggregation")

        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated_dict)
        if self.processed_algorithm is not None:
            dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, self.processed_algorithm)
            self.processed_algorithm = None

        return agg_client["contributor_name"],dxo
