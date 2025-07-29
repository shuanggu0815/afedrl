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
import os
import json
import torch

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms 
from torchvision import models 
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from trainer.unet import UNet

from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
)
from utils.custom_client_datalist_json_path import custom_client_datalist_json_path

class Cifar10Validator(Executor):
    def __init__(self, data_path="afedrl/prostate/med_evi_debug/app/config/config_train.json", validate_task_name=AppConstants.TASK_VALIDATION):
        super().__init__()

        self._validate_task_name = validate_task_name
        self.data_path = data_path
        self.test_loader = None

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = UNet(
            in_channels=1,
            out_channels=1,
        ).to(self.device)

        self.transform = None
        self.transform_post = None
        self.inferer = None
        self.valid_metric = None

        self.is_initialized = False


    def initialize(self, fl_ctx: FLContext):
        if not os.path.isfile(self.data_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {self.data_path}",
            )

        with open(self.data_path) as file:
            self.config_info = json.load(file)
        
        cache_rate = self.config_info["cache_dataset"]
        dataset_base_dir = self.config_info["dataset_base_dir"]
        datalist_json_path = self.config_info["datalist_json_path"]
         # Get datalist json
        datalist_json_path = custom_client_datalist_json_path(datalist_json_path, self.client_id)
        test_list = load_decathlon_datalist(
            data_list_file_path=datalist_json_path,
            is_segmentation=True,
            data_list_key="testing",
            base_dir=dataset_base_dir,
        )

        self.transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(keys=["image", "label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
                Resized(
                    keys=["image", "label"],
                    spatial_size=(256, 256),
                    mode=("bilinear"),
                    align_corners=True,
                ),
                AsDiscreted(keys=["label"], threshold=0.5),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.transform_post = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        if cache_rate > 0.0:
            test_dataset = CacheDataset(
                data=test_list,
                transform=self.transform,
                cache_rate=cache_rate,
                num_workers=0,
            )
        else:
            test_dataset = Dataset(
                data=test_list,
                transform=self.transform,
            )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        self.inferer = SimpleInferer()
        self.valid_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")
        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)

        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self._validate(weights, fl_ctx, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, fl_ctx, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()
    
        with torch.no_grad():
            metric = 0
            for i, batch_data in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                test_images = batch_data["image"].to(self.device)
                test_labels = batch_data["label"].to(self.device)
                # Inference
                test_outputs = self.inferer(test_images, self.model)
                test_outputs = self.transform_post(test_outputs)
                # Compute metric
                metric_score = self.valid_metric(y_pred=test_outputs, y=test_labels)
                metric += metric_score.item()
            # compute mean dice over whole validation set
            metric /= len(self.test_loader)
        
        return metric
