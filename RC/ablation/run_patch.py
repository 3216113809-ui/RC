"""
RCIL run.py 集成指南 —— 精简版 (3 处修改)
=============================================

第 1 处: argparse 区域添加参数
--------------------------------
parser.add_argument('--ablation_exp', type=str, default=None,
                    help='Ablation experiment name, e.g. no_rc, no_pcd, no_frozen')
parser.add_argument('--ablation_save_dir', type=str, default='./ablation_results')


第 2 处: 模型构建后 (约在 model = build_model(args) 之后)
--------------------------------
if args.ablation_exp:
    from ablation.integrate import apply_ablation
    model = apply_ablation(args, model)


第 3 处: 训练循环中, 每步结束后
--------------------------------
# 推理/评估前:
from ablation.integrate import post_step
post_step(model)

# 下一步训练前:
from ablation.integrate import pre_step
pre_step(model)


完成后, 运行:
  python run.py --dataset voc --setting 15-1 --ablation_exp no_rc
"""

if __name__ == "__main__":
    print(__doc__)
