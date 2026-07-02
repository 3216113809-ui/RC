"""
消融实验配置 —— 仅保留有清晰对照关系的实验

每组实验只改变一个变量，其余完全相同:
  A. RC模块对照:     rcil_full  vs  no_rc
  B. PCD蒸馏对照:    rcil_full  vs  no_pcd
  C. 冻结分支对照:   rcil_full  vs  no_frozen
  D. Drop-path对照:  η=0.0  vs  η=0.5  vs  η=1.0
  E. PCD模式对照:    skd_only  vs  ckd_only  vs  rcil_full
  F. 下界对照:       finetune  (无RC无蒸馏, 纯微调)
"""

# ============================================================
# 基准配置 (rcil_full)
# ============================================================
_BASE = {
    "use_rc_module": True,
    "rc_frozen_branch": True,
    "use_pcd_distillation": True,
    "pcd_mode": "both",
    "drop_path_rate": 0.5,
    "loss_weights": {"unce": 1.0, "unkd": 10.0, "skd": 1.0, "ckd": 1.0},
    "pcd_kernel_sizes": [4, 8, 12, 16, 20, 24],
}

# ============================================================
# A. RC 模块对照 (改变: use_rc_module)
# ============================================================
#   rcil_full  ← RC + PCD
#   no_rc      ← 标准卷积 + PCD  (其余参数完全相同)
#   → 回答: RC 模块带来多少增益?
# ============================================================
ABLATION_RC = {
    "rcil_full": {
        **_BASE,
        "description": "RC + PCD (full method, reference)",
    },
    "no_rc": {
        **_BASE,
        "use_rc_module": False,
        "description": "-RC: standard conv + PCD, tests RC module contribution",
    },
}

# ============================================================
# B. PCD 蒸馏对照 (改变: use_pcd_distillation)
# ============================================================
#   rcil_full  ← RC + PCD
#   no_pcd     ← RC only, 无任何蒸馏  (其余参数完全相同)
#   → 回答: PCD 蒸馏带来多少增益?
# ============================================================
ABLATION_PCD = {
    "rcil_full": {
        **_BASE,
        "description": "RC + PCD (full method, reference)",
    },
    "no_pcd": {
        **_BASE,
        "use_pcd_distillation": False,
        "loss_weights": {"unce": 1.0, "unkd": 0.0, "skd": 0.0, "ckd": 0.0},
        "description": "-PCD: RC only, no distillation, tests PCD contribution",
    },
}

# ============================================================
# C. 冻结分支对照 (改变: rc_frozen_branch)
# ============================================================
#   rcil_full   ← 有冻结分支
#   no_frozen   ← 无冻结分支 (RC退化为可训练卷积 + drop-path)
#   → 回答: 冻结分支(旧知识保留机制)带来多少增益?
# ============================================================
ABLATION_FROZEN = {
    "rcil_full": {
        **_BASE,
        "description": "RC + PCD + frozen branch (full method, reference)",
    },
    "no_frozen": {
        **_BASE,
        "rc_frozen_branch": False,
        "drop_path_rate": 0.0,  # 无冻结分支则 drop-path 无意义
        "description": "-Frozen: RC without frozen branch, tests old-knowledge preservation",
    },
}

# ============================================================
# D. Drop-path rate 对照 (改变: drop_path_rate)
# ============================================================
#   η=0.0  ← 始终用冻结分支输出 (完全保留旧知识)
#   η=0.5  ← 默认, 随机门控
#   η=1.0  ← 始终用可训练分支输出 (完全偏向新知识)
#   → 回答: 新旧知识平衡策略的影响?
# ============================================================
ABLATION_DROPPATH = {
    "droppath_0.0": {
        **_BASE,
        "drop_path_rate": 0.0,
        "description": "drop-path η=0.0: always keep frozen branch",
    },
    "droppath_0.5": {
        **_BASE,
        "drop_path_rate": 0.5,
        "description": "drop-path η=0.5: default stochastic gate",
    },
    "droppath_1.0": {
        **_BASE,
        "drop_path_rate": 1.0,
        "description": "drop-path η=1.0: always use trainable branch",
    },
}

# ============================================================
# E. PCD 模式对照 (改变: pcd_mode)
# ============================================================
#   skd_only   ← 仅空间 PCD
#   ckd_only   ← 仅通道 PCD
#   rcil_full  ← 空间+通道 PCD
#   → 回答: SKD 和 CKD 各自贡献多少? 两者叠加是否互补?
# ============================================================
ABLATION_PCD_MODE = {
    "skd_only": {
        **_BASE,
        "pcd_mode": "spatial_only",
        "loss_weights": {"unce": 1.0, "unkd": 10.0, "skd": 1.0, "ckd": 0.0},
        "description": "Spatial PCD only, tests SKD contribution",
    },
    "ckd_only": {
        **_BASE,
        "pcd_mode": "channel_only",
        "loss_weights": {"unce": 1.0, "unkd": 10.0, "skd": 0.0, "ckd": 1.0},
        "description": "Channel PCD only, tests CKD contribution",
    },
    "rcil_full": {
        **_BASE,
        "description": "Spatial + Channel PCD (full method, reference)",
    },
}

# ============================================================
# F. 下界对照
# ============================================================
#   finetune  ← 无 RC, 无蒸馏, 纯微调
#   rcil_full ← 完整方法
#   → 回答: 完整方法相比纯微调提升多少?
# ============================================================
ABLATION_LOWER_BOUND = {
    "finetune": {
        "use_rc_module": False,
        "use_pcd_distillation": False,
        "rc_frozen_branch": False,
        "drop_path_rate": 0.0,
        "loss_weights": {"unce": 1.0, "unkd": 0.0, "skd": 0.0, "ckd": 0.0},
        "description": "Fine-tune only: no RC, no KD (lower bound)",
    },
    "rcil_full": {
        **_BASE,
        "description": "RCIL full method (upper bound)",
    },
}

# ============================================================
# 汇总: 所有可对照实验
# ============================================================
ALL_ABLATIONS = {}
ALL_ABLATIONS.update(ABLATION_RC)
ALL_ABLATIONS.update({k: v for k, v in ABLATION_PCD.items() if k != "rcil_full"})
ALL_ABLATIONS.update({k: v for k, v in ABLATION_FROZEN.items() if k != "rcil_full"})
ALL_ABLATIONS.update(ABLATION_DROPPATH)
ALL_ABLATIONS.update({k: v for k, v in ABLATION_PCD_MODE.items() if k != "rcil_full"})
ALL_ABLATIONS.update({k: v for k, v in ABLATION_LOWER_BOUND.items() if k != "rcil_full"})

# ============================================================
# 对照关系表 (用于报告生成)
# ============================================================
COMPARISON_PAIRS = [
    {
        "name": "A. RC Module",
        "question": "RC 表示补偿模块的贡献?",
        "control": "rcil_full",
        "variable": "no_rc",
        "delta": "use_rc_module: True → False",
    },
    {
        "name": "B. PCD Distillation",
        "question": "PCD 池化立方体蒸馏的贡献?",
        "control": "rcil_full",
        "variable": "no_pcd",
        "delta": "PCD → no distillation",
    },
    {
        "name": "C. Frozen Branch",
        "question": "冻结分支(旧知识保留)的贡献?",
        "control": "rcil_full",
        "variable": "no_frozen",
        "delta": "rc_frozen_branch: True → False",
    },
    {
        "name": "D. Drop-path Rate",
        "question": "新旧知识平衡门控的影响?",
        "control": None,  # 三向对照
        "variable": None,
        "delta": "η ∈ {0.0, 0.5, 1.0}",
        "group": ["droppath_0.0", "droppath_0.5", "droppath_1.0"],
    },
    {
        "name": "E. PCD Mode",
        "question": "SKD vs CKD 各自及组合的贡献?",
        "control": None,  # 三向对照
        "variable": None,
        "delta": "spatial / channel / both",
        "group": ["skd_only", "ckd_only", "rcil_full"],
    },
    {
        "name": "F. Overall Gain",
        "question": "完整方法相比纯微调提升多少?",
        "control": "finetune",
        "variable": "rcil_full",
        "delta": "no RC/KDL → full RCIL",
    },
]
