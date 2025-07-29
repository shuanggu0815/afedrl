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

import copy
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs

from utils.autofedasync_constants import AutoFedRLConstants
from utils.custom_client_datalist_json_path import custom_client_datalist_json_path
from trainer.cifar10_learner import CIFAR10Learner
# from monai.networks.nets.unet import UNet
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

class CIFAR10AutoFedRLearner(CIFAR10Learner):  # TODO: also support CIFAR10ScaffoldLearner
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        train_config_filename: str = "config/config_train.json",
        aggregation_epochs: int = 1,  # TODO: Is this still being used?
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
        entropy_coeff: float = 1.0,
        entropy_threshold: float = 2.0,
    ):
        """Simple CIFAR-10 Trainer utilizing Auto-FedRL.

        Args:
            train_idx_root: directory with site training indices for CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.
            entropy_coeff:  entropy cut-off.
            entropy_threshold:  entropy threshold.

        Returns:
            a Shareable with the updated local model after running `train()`,
            or validation metrics after calling `validate()`,
            or the best local model when calling `get_model_for_validation()`
        """

        CIFAR10Learner.__init__(
            self,
            train_idx_root=train_idx_root,
            aggregation_epochs=aggregation_epochs,
            lr=lr,
            fedproxloss_mu=fedproxloss_mu,
            central=central,
            analytic_sender_id=analytic_sender_id,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_config_filename = train_config_filename
        self.config_info = None
        self.entropy_coeff = entropy_coeff
        self.entropy_threshold = entropy_threshold
        self.transform = None
        self.transform_post = None
        self.weight = 0

        self.current_round = 0
        self.best_global_acc = 0

        # Use FOBS serializing/deserializing PyTorch tensors
        fobs.register(TensorDecomposer)

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # Initialize super class
        CIFAR10Learner.initialize(self, parts=parts, fl_ctx=fl_ctx)
        # Enabling the Nesterov momentum can stabilize the training.
        train_config_file_path = os.path.join(self.app_root, self.train_config_filename)
        if not os.path.isfile(train_config_file_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {train_config_file_path}",
            )
        with open(train_config_file_path) as file:
            self.config_info = json.load(file)
        
        batch_size = self.config_info["batch_size"]
        cache_rate = self.config_info["cache_dataset"]
        dataset_base_dir = self.config_info["dataset_base_dir"]
        datalist_json_path = self.config_info["datalist_json_path"]
         # Get datalist json
        datalist_json_path = custom_client_datalist_json_path(datalist_json_path, self.client_id)

        # Set datalist
        train_list = load_decathlon_datalist(
            data_list_file_path=datalist_json_path,
            is_segmentation=True,
            data_list_key="training",
            base_dir=dataset_base_dir,
        )
        valid_list = load_decathlon_datalist(
            data_list_file_path=datalist_json_path,
            is_segmentation=True,
            data_list_key="validation",
            base_dir=dataset_base_dir,
        )
        test_list = load_decathlon_datalist(
            data_list_file_path=datalist_json_path,
            is_segmentation=True,
            data_list_key="testing",
            base_dir=dataset_base_dir,
        )
        self.log_info(
            fl_ctx,
            f"Training Size: {len(train_list)}, Validation Size: {len(valid_list)}",
        )

        self.weight = len(train_list)/1967.0

        # Set the training-related context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UNet(
            in_channels=1,
            out_channels=2,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99))
        self.criterion = DiceLoss(softmax=True,include_background=False)

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
        self.transform_post = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True)])

        # Set dataset
        if cache_rate > 0.0:
            train_dataset = CacheDataset(
                data=train_list,
                transform=self.transform,
                cache_rate=cache_rate,
                num_workers=0,
            )
            valid_dataset = CacheDataset(
                data=valid_list,
                transform=self.transform,
                cache_rate=cache_rate,
                num_workers=0,
            )
            test_dataset = CacheDataset(
                data=test_list,
                transform=self.transform,
                cache_rate=cache_rate,
                num_workers=0,
            )
        else:
            train_dataset = Dataset(
                data=train_list,
                transform=self.transform,
            )
            valid_dataset = Dataset(
                data=valid_list,
                transform=self.transform,
            )
            test_dataset = CacheDataset(
                data=test_list,
                transform=self.transform,
                )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


        # Set inferer and evaluation metric
        self.inferer = SimpleInferer()
        self.valid_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)


    def _get_model_weights(self,fl_ctx) -> Shareable:
        # Get the new state dict and send as weights

        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        n_iterations = len(self.train_loader)
       
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: n_iterations}
        )
        return outgoing_dxo.to_shareable()
    
    def local_train(self, fl_ctx, train_loader, model_global, abort_signal: Signal, val_freq: int = 0):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch +1
            self.log_info(fl_ctx, f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs}")
            avg_loss = 0.0
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                inverted_labels = 1 - labels  # 1-0=1，1-1=0
                labels_new = torch.cat((inverted_labels, labels), dim=1)
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_new)


                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Backward + Optimize
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            self.writer.add_scalar("train_loss", avg_loss / len(train_loader), self.epoch_global)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.test_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if fl_ctx.get_identity_name() == "client_MSD":
            time.sleep(50)
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()} sleep 50s")
        if fl_ctx.get_identity_name() == "client_Promise12":
            time.sleep(50)
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()} sleep 50s")
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        self.current_round = current_round

        # Get lr and ne from server
        current_lr, current_ne = None, None
        self.log_info(fl_ctx, f"shareble get_header:{shareable.get_header(AutoFedRLConstants.GLOBAL_HYPERPARAMTER_COLLECTION)}")
        hps = fobs.loads(shareable.get_header(AutoFedRLConstants.GLOBAL_HYPERPARAMTER_COLLECTION))
        c_hp = fl_ctx.get_prop(AutoFedRLConstants.CLIENT_HYPERPARAMTER_COLLECTION)
        self.log_info(fl_ctx,f"client_hp:{c_hp}")
        if hps is not None:
            current_lr = hps.get("lr")
        if c_hp is not None:
            current_ne = c_hp.get("ne")
            aw = c_hp.get("aw")
            # aw = 0.5
            current_lr = c_hp.get("lr")
        if current_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            self.log_info(fl_ctx, f"Received and override current learning rate as: {current_lr}")
        if current_ne is not None:
            # site_name = fl_ctx.get_identity_name()
            # site_index = int(site_name.split('-')[-1]) - 1 
            # self.aggregation_epochs = current_ne[site_index] if 0 <= site_index < len(current_ne) else None
            self.aggregation_epochs = current_ne
            self.log_info(fl_ctx, f"Received and override current number of local epochs: {current_ne}")

        # Update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # Reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # Update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed!") from e
        self.model.load_state_dict(local_var_dict)

        # Local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # Make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # Local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
            val_freq=1 if self.central else 0,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # Perform valid after local train
        acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")

        # Save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        # Compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        # model_diff = {}
        # for name in global_weights:
        #     if name not in local_weights:
        #         continue
        #     model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
        #     if np.any(np.isnan(model_diff[name])):
        #         self.system_panic(f"{name} weights became NaN...", fl_ctx)
        #         return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Build the shareable
        result = {
            "local_weights":local_weights,
            "aw":aw
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=result)
        if c_hp.get("aw") is not None:
            # When search aggregation weights, we have to override it
            # to 1, since we will manually assign weights to aggregator.
            # Search space will discover which client is more informative.
            # It might not be related to the number of data in a client.
            dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)
        else:
            dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None, get_loss=False):
        self.model.eval()
        with torch.no_grad():
            metric, total = 0, 0
            for _, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                inverted_labels = 1 - labels  # 1-0=1，1-1=0
                labels_new = torch.cat((inverted_labels, labels), dim=1)
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_new)

                if get_loss:
                    # Return val loss instead of accuracy over the number batches
                    total += inputs.data.size()[0]
                    metric += loss.item()
                else:
                    total += inputs.data.size()[0]
                    val_outputs = self.inferer(inputs, self.model)
                    val_outputs_l = torch.stack(list(map(lambda x: self.transform_post(x), val_outputs)))
                    metric_score = self.valid_metric(y_pred=val_outputs_l, y=labels)
                    metric += metric_score.mean().item()
        
            if get_loss:
                metric = metric / float(len(valid_loader))
                self.log_info(fl_ctx, f"HP Search loss: {metric} of {total} batches on {fl_ctx.get_identity_name()}")
            else:
                metric = metric / float(len(valid_loader))
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.current_round)
        return metric
    
    def validate_for_search(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            # Evaluating global model during training
            model_owner = "global_model"

        # Update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        global_model = copy.deepcopy(self.model)

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        global_var_dict = global_model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0

        for var_name in global_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    global_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, global_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight for {var_name} failed!") from e


        global_model.load_state_dict(global_var_dict)

        if n_loaded == 0:
            raise ValueError(f"No weights loaded for global model! Received weight dict is {global_weights}")

        self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()} for HP Search")

        global_model.eval()
        with torch.no_grad():
            metric, total = 0, 0
            total_loss = 0
            for _, batch_data in enumerate(self.valid_loader):
                if abort_signal.triggered:
                    return None
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                inverted_labels = 1 - labels  # 1-0=1，1-1=0
                labels_new = torch.cat((inverted_labels, labels), dim=1)
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_new)
                 # Inference
                val_outputs = self.inferer(inputs, global_model)
                val_outputs_l = torch.stack(list(map(lambda x: self.transform_post(x), val_outputs)))
                # Compute metric
                metric_score = self.valid_metric(y_pred=val_outputs_l, y=labels)
                metric += metric_score.mean().item()

                total += inputs.data.size()[0]
                total_loss += loss.item()
                    
          
            val_loss_hp  = total_loss / float(len(self.valid_loader))
            acc = metric / len(self.valid_loader)
            self.log_info(fl_ctx, f"HP Search loss: {val_loss_hp } of {total} batches on {fl_ctx.get_identity_name() }"
                              f"acc:{acc}")
            
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        val_results = {"val_loss": val_loss_hp}

        metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
        return metric_dxo.to_shareable()
    
    def validate_and_score(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            # Evaluating global model during training
            model_owner = "global_model"

        # receive global model
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        global_model = copy.deepcopy(self.model)

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        global_var_dict = global_model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0

        for var_name in global_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    global_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, global_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight for {var_name} failed!") from e


        global_model.load_state_dict(global_var_dict)

        if n_loaded == 0:
            raise ValueError(f"No weights loaded for global model! Received weight dict is {global_weights}")

        self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()} for HP Search")

        global_model.eval()
        self.model.eval()
        udis_list = torch.tensor([]).cuda()
        udata_list = torch.tensor([]).cuda()
        with torch.no_grad():
            metric, total = 0, 0
            total_loss = 0
            for _, batch_data in enumerate(self.valid_loader):
                if abort_signal.triggered:
                    return None
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                inverted_labels = 1 - labels  # 1-0=1，1-1=0
                labels_new = torch.cat((inverted_labels, labels), dim=1)
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_new)
                
                 # Inference
                val_outputs = self.inferer(inputs, global_model)
                val_outputs_l = torch.stack(list(map(lambda x: self.transform_post(x), val_outputs)))

                total_loss += loss.item()
                metric_score = self.valid_metric(y_pred=val_outputs_l, y=labels)
                metric += metric_score.mean().item()


                g_logit = torch.clamp_max(outputs, 80)
                alpha = torch.exp(g_logit) + 1
                total_alpha = torch.sum(alpha, dim=1, keepdim=True) # batch_size, 1, patch_size, patch_size
                g_pred = alpha / total_alpha
                g_entropy = torch.sum(- g_pred * torch.log(g_pred), dim=1)     
                g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            
                g_u_dis = g_entropy - g_u_data
                udis_list = torch.cat((udis_list, g_u_dis.mean(dim=[1,2])))

                # local model
                l_logit = self.model(inputs)
                l_logit = torch.clamp_max(l_logit, 80)
                alpha = torch.exp(l_logit) + 1
                print(f"alpha shape:{alpha.shape}")
                print(f"alpha :{alpha.sum()}")
                total_alpha = torch.sum(alpha, dim=1, keepdim=True) # batch_size, 1, patch_size, patch_size
                print(f"total_alpha shape:{total_alpha.shape}")
                l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
                print(f"l_u_data shape:{l_u_data.shape}")
                print(f"l_u_data :{l_u_data.sum()}")
                udata_list = torch.cat((udata_list, l_u_data.mean(dim=[1,2])))
                    
          
            val_loss_hp  = total_loss / float(len(self.valid_loader))
            acc = metric / float(len(self.valid_loader))
            self.log_info(fl_ctx, f"HP Search loss: {val_loss_hp } of {total} batches on {fl_ctx.get_identity_name() }"
                              f"acc:{acc}")
            
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        udis = udis_list.mean().cpu().numpy()
        udata = udata_list.mean().cpu().numpy()
        val_loss_hp = val_loss_hp * self.weight
        val_results = {"val_loss": val_loss_hp,
                        "udis": udis,
                        "udata":udata}

        metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
        return metric_dxo.to_shareable()


    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            # Evaluating global model during training
            model_owner = "global_model"

        # Update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # Update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed!") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # Perform valid before local train
            global_acc = self.local_valid(self.test_loader, abort_signal, tb_id="val_acc_global_model", fl_ctx=fl_ctx)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_acc_global_model ({model_owner}): {global_acc}")

            if global_acc > self.best_global_acc:
                self.best_global_acc = global_acc
            # Log the best global model_accuracy
            self.writer.add_scalar("best_val_acc_global_model", self.best_global_acc, self.current_round)

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: global_acc}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # Perform valid
            train_acc = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training acc ({model_owner}): {train_acc}")

            val_acc = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc ({model_owner}): {val_acc}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        elif validate_type == AutoFedRLConstants.MODEL_VALIDATE_FOR_SEARCH:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()} for HP Search")

            val_loss_hp = self.local_valid(self.valid_loader, abort_signal, fl_ctx=fl_ctx, get_loss=True)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            val_results = {"val_loss": val_loss_hp}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
