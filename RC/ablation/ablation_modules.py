"""
消融实验核心模块 —— 仅保留对照实验需要的变体

RC 模块变体:
  - RCConv2d         → 完整双分支 RC (rcil_full 及其参数变体用)
  - RCConv2d_NoFrozen → 无冻结分支 (no_frozen 实验用)
  - StandardConv2d    → 标准卷积 (no_rc 实验用)

PCD 蒸馏变体:
  - PooledCubeDistillation → 完整 PCD (通过 pcd_mode 切换 spatial/channel/both)
  - (no_pcd 通过设置 use_pcd_distillation=False 实现, 不需要单独模块)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RC 卷积模块变体
# ============================================================

class RCConv2d(nn.Module):
    """
    完整 RC 卷积 (论文默认)

    训练: y = η · W_frozen * x + (1-η) · W_trainable * x  (通道级随机门控)
    推理: y = (W_frozen + W_trainable) * x  (结构重参数化)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1,
                 bias=False, drop_path_rate=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.drop_path_rate = drop_path_rate

        self.frozen_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        for p in self.frozen_conv.parameters():
            p.requires_grad = False

        self.train_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

        self._merged = False

    def _drop_mask(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return torch.ones(self.out_channels, device=x.device)
        keep = torch.bernoulli(
            torch.full((self.out_channels,), self.drop_path_rate, device=x.device))
        return keep.view(1, -1, 1, 1)

    def forward(self, x):
        if self._merged:
            return self.merged_conv(x)
        out_f = self.frozen_conv(x)
        out_t = self.train_conv(x)
        if self.training and self.drop_path_rate > 0.0:
            mask = self._drop_mask(x)
            return mask * out_f + (1 - mask) * out_t
        return out_f + out_t

    def merge(self):
        """合并双分支 → 单卷积分支, 零推理开销"""
        if self._merged:
            return
        self.merged_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
            bias=(self.frozen_conv.bias is not None))
        self.merged_conv.weight.data = (
            self.frozen_conv.weight.data + self.train_conv.weight.data)
        if self.frozen_conv.bias is not None:
            self.merged_conv.bias.data = (
                self.frozen_conv.bias.data + self.train_conv.bias.data)
        self._merged = True

    def unmerge(self):
        """取消合并, 恢复双分支 (下一步增量训练前调用)"""
        if not self._merged:
            return
        self.train_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
            bias=(self.frozen_conv.bias is not None)).to(
                self.merged_conv.weight.device)
        self.train_conv.weight.data.copy_(self.merged_conv.weight.data)
        if self.frozen_conv.bias is not None:
            self.train_conv.bias.data.copy_(self.merged_conv.bias.data)
        del self.merged_conv
        self._merged = False


class RCConv2d_NoFrozen(RCConv2d):
    """
    Ablation: 去掉冻结分支 (no_frozen 实验用)
    退化为: y = W_trainable * x, 无旧知识保留
    """

    def __init__(self, *args, **kwargs):
        kwargs["drop_path_rate"] = 0.0
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self._merged:
            return self.merged_conv(x)
        return self.train_conv(x)

    def merge(self):
        if self._merged:
            return
        self.merged_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
            bias=(self.frozen_conv.bias is not None))
        self.merged_conv.weight.data = self.train_conv.weight.data.clone()
        if self.frozen_conv.bias is not None:
            self.merged_conv.bias.data = self.train_conv.bias.data.clone()
        self._merged = True


class StandardConv2d(nn.Conv2d):
    """Ablation: 标准卷积 (no_rc 实验用)"""
    pass


# ============================================================
# PCD 蒸馏模块
# ============================================================

class PooledCubeDistillation(nn.Module):
    """
    Pooled Cube Distillation

    通过 pcd_mode 切换:
      - "spatial_only": 仅在空间维度池化后蒸馏 (SKD)
      - "channel_only": 仅在通道维度池化后蒸馏 (CKD)
      - "both":         两者都做 (完整 PCD)
    """

    def __init__(self, pcd_mode="both", kernel_sizes=None,
                 skd_weight=1.0, ckd_weight=1.0, temperature=1.0):
        super().__init__()
        self.pcd_mode = pcd_mode
        self.kernel_sizes = kernel_sizes or [4, 8, 12, 16, 20, 24]
        self.skd_weight = skd_weight
        self.ckd_weight = ckd_weight
        self.temperature = temperature

    def _multi_scale_pool(self, feat, ks):
        """对特征做 kernel_size=ks 的自适应平均池化"""
        B, C, H, W = feat.shape
        out_h, out_w = min(ks, H), min(ks, W)
        p = F.adaptive_avg_pool2d(feat, (out_h, out_w))
        if out_h != ks or out_w != ks:
            p = F.interpolate(p, size=(ks, ks), mode="bilinear", align_corners=False)
        return p

    def spatial_kd(self, feat_s, feat_t):
        """空间 PCD: 多尺度池化后逐像素 MSE"""
        loss = 0.0
        for ks in self.kernel_sizes:
            ps = self._multi_scale_pool(feat_s, ks)
            pt = self._multi_scale_pool(feat_t, ks)
            loss += F.mse_loss(ps, pt)
        return self.skd_weight * loss / len(self.kernel_sizes) / (self.temperature ** 2)

    def channel_kd(self, feat_s, feat_t):
        """通道 PCD: 全局池化后逐通道 MSE"""
        loss = 0.0
        B, C = feat_s.shape[:2]
        for ks in self.kernel_sizes:
            cs = F.adaptive_avg_pool2d(feat_s, (1, 1)).view(B, C)
            ct = F.adaptive_avg_pool2d(feat_t, (1, 1)).view(B, C)
            loss += F.mse_loss(cs, ct)
        return self.ckd_weight * loss / len(self.kernel_sizes) / (self.temperature ** 2)

    def forward(self, feat_s_list, feat_t_list):
        """
        Args:
            feat_s_list: student 特征列表
            feat_t_list: teacher 特征列表
        Returns:
            {"skd_loss": ..., "ckd_loss": ..., "total": ...}
        """
        skd = torch.tensor(0.0, device=feat_s_list[0].device)
        ckd = torch.tensor(0.0, device=feat_s_list[0].device)

        for fs, ft in zip(feat_s_list, feat_t_list):
            if self.pcd_mode in ("spatial_only", "both"):
                skd += self.spatial_kd(fs, ft)
            if self.pcd_mode in ("channel_only", "both"):
                ckd += self.channel_kd(fs, ft)

        n = max(len(feat_s_list), 1)
        return {"skd_loss": skd / n, "ckd_loss": ckd / n, "total": (skd + ckd) / n}
