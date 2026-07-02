"""
与原始 run.py 的集成桥接 (最小侵入)

在 run.py 中只需添加 3 处代码:

  [1] argparse 添加:
      parser.add_argument('--ablation_exp', type=str, default=None)

  [2] 模型构建后:
      if args.ablation_exp:
          from ablation.integrate import apply_ablation
          model = apply_ablation(args, model)

  [3] 每步训练结束 / 下步开始前:
      from ablation.integrate import post_step, pre_step
      post_step(model)   # 推理前合并 RC 分支
      pre_step(model)    # 训练前恢复双分支
"""

import os
import json

from .ablation_configs import ALL_ABLATIONS
from .ablation_trainer import build_ablation_model, merge_all_rc, unmerge_all_rc


def apply_ablation(args, model):
    """
    根据 --ablation_exp 修改模型

    Args:
        args:  命令行参数 (需含 ablation_exp)
        model: RCIL 模型实例
    Returns:
        修改后的模型
    """
    exp = args.ablation_exp
    if exp not in ALL_ABLATIONS:
        raise ValueError(f"Unknown ablation: {exp}. Choices: {list(ALL_ABLATIONS.keys())}")

    cfg = ALL_ABLATIONS[exp]
    print(f"[Ablation] {exp}: {cfg['description']}")
    model = build_ablation_model(model, cfg)

    # 保存配置
    save_dir = getattr(args, 'ablation_save_dir', './ablation_results')
    os.makedirs(os.path.join(save_dir, exp), exist_ok=True)
    with open(os.path.join(save_dir, exp, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    args.ablation_config = cfg
    return model


def post_step(model):
    """每个增量步骤训练结束后: 合并 RC 双分支用于推理"""
    merge_all_rc(model)


def pre_step(model):
    """每个增量步骤训练开始前: 恢复 RC 双分支"""
    unmerge_all_rc(model)
