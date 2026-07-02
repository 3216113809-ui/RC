"""
消融实验训练器

每个实验只改变一个变量, 其余参数与 rcil_full 完全相同,
从而形成清晰的对照关系。
"""

import os
import json
import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from .ablation_modules import RCConv2d, RCConv2d_NoFrozen, StandardConv2d, PooledCubeDistillation


def build_ablation_model(model, exp_config):
    """
    根据消融配置修改模型 —— 只改需要对照的变量

    Args:
        model:      原始 RCIL 模型
        exp_config: 消融配置字典

    Returns:
        修改后的模型
    """
    use_rc = exp_config.get("use_rc_module", True)
    rc_frozen = exp_config.get("rc_frozen_branch", True)
    drop_path_rate = exp_config.get("drop_path_rate", 0.5)
    use_pcd = exp_config.get("use_pcd_distillation", True)

    # --- RC 模块替换 ---
    if not use_rc:
        _replace_rc_with(model, StandardConv2d)
    elif not rc_frozen:
        _replace_rc_with(model, RCConv2d_NoFrozen)
    else:
        _set_drop_path(model, drop_path_rate)

    # --- PCD 蒸馏配置 ---
    if use_pcd:
        model.distill_module = PooledCubeDistillation(
            pcd_mode=exp_config.get("pcd_mode", "both"),
            kernel_sizes=exp_config.get("pcd_kernel_sizes", [4, 8, 12, 16, 20, 24]),
            skd_weight=exp_config.get("loss_weights", {}).get("skd", 1.0),
            ckd_weight=exp_config.get("loss_weights", {}).get("ckd", 1.0),
        )
    else:
        model.distill_module = None

    model.ablation_config = exp_config
    return model


def _replace_rc_with(model, target_cls):
    """递归替换所有 RCConv2d → target_cls"""
    for name, module in model.named_children():
        if isinstance(module, RCConv2d):
            new_mod = _rc_to_target(module, target_cls)
            setattr(model, name, new_mod)
        else:
            _replace_rc_with(module, target_cls)


def _rc_to_target(rc_mod, target_cls):
    """单个 RCConv2d → target_cls 的权重迁移"""
    kwargs = dict(
        in_channels=rc_mod.in_channels, out_channels=rc_mod.out_channels,
        kernel_size=rc_mod.kernel_size, stride=rc_mod.stride,
        padding=rc_mod.padding, dilation=rc_mod.dilation,
        groups=rc_mod.groups, bias=(rc_mod.frozen_conv.bias is not None))
    if target_cls == StandardConv2d:
        new_mod = target_cls(**kwargs)
        new_mod.weight.data.copy_(rc_mod.train_conv.weight.data)
        if new_mod.bias is not None:
            new_mod.bias.data.copy_(rc_mod.train_conv.bias.data)
        return new_mod
    elif target_cls == RCConv2d_NoFrozen:
        return target_cls(**kwargs)
    return target_cls(**kwargs)


def _set_drop_path(model, rate):
    """修改所有 RCConv2d 的 drop_path_rate"""
    for m in model.modules():
        if isinstance(m, RCConv2d):
            m.drop_path_rate = rate


def merge_all_rc(model):
    """推理前合并所有 RC 双分支"""
    for m in model.modules():
        if isinstance(m, RCConv2d):
            m.merge()


def unmerge_all_rc(model):
    """下一步训练前恢复双分支"""
    for m in model.modules():
        if isinstance(m, RCConv2d):
            m.unmerge()


def copy_model_for_distill(model):
    """深拷贝模型用于蒸馏 (冻结参数)"""
    old = copy.deepcopy(model)
    old.eval()
    for p in old.parameters():
        p.requires_grad = False
    return old


class AblationRunner:
    """管理单个消融实验的完整流程"""

    def __init__(self, exp_name, exp_config, save_dir="./ablation_results", seed=42):
        self.exp_name = exp_name
        self.exp_config = exp_config
        self.save_dir = os.path.join(save_dir, exp_name)
        self.seed = seed
        self.history = []
        os.makedirs(self.save_dir, exist_ok=True)

    def setup(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(self.exp_config, f, indent=2)
        print(f"[{self.exp_name}] {self.exp_config['description']}")

    def train_step(self, model, loader, optimizer, criterion, step_idx):
        """单步增量训练"""
        model.train()
        losses = OrderedDict(total=0.0, unce=0.0, unkd=0.0, skd=0.0, ckd=0.0)

        old_model = None
        if step_idx > 0 and model.distill_module is not None:
            old_model = copy_model_for_distill(model)

        lw = self.exp_config.get("loss_weights", {})

        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs, features = model(images, return_features=True)
            loss = lw.get("unce", 1.0) * nn.functional.cross_entropy(outputs, labels, ignore_index=255)
            losses["unce"] += loss.item()

            if old_model is not None:
                with torch.no_grad():
                    _, old_feats = old_model(images, return_features=True)
                    old_out = old_model(images)

                # logit-level KD
                if lw.get("unkd", 0) > 0:
                    loss_unkd = nn.functional.kl_div(
                        nn.functional.log_softmax(outputs, dim=1),
                        nn.functional.softmax(old_out, dim=1),
                        reduction="batchmean")
                    loss += lw["unkd"] * loss_unkd
                    losses["unkd"] += loss_unkd.item()

                # PCD
                if model.distill_module is not None:
                    kd = model.distill_module(features, old_feats)
                    if lw.get("skd", 0) > 0:
                        loss += lw["skd"] * kd["skd_loss"]
                    if lw.get("ckd", 0) > 0:
                        loss += lw["ckd"] * kd["ckd_loss"]
                    losses["skd"] += kd["skd_loss"].item()
                    losses["ckd"] += kd["ckd_loss"].item()

            loss.backward()
            optimizer.step()

        n = max(len(loader), 1)
        return {k: v / n for k, v in losses.items()}

    def save_results(self, metrics, final_miou):
        data = {"exp_name": self.exp_name, "description": self.exp_config["description"],
                "final_mIoU": final_miou, "metrics": metrics}
        with open(os.path.join(self.save_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)
