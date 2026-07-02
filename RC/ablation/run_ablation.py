"""
消融实验主入口

6 组对照实验, 每组只改变一个变量:

  A. RC模块:    rcil_full  vs  no_rc
  B. PCD蒸馏:   rcil_full  vs  no_pcd
  C. 冻结分支:  rcil_full  vs  no_frozen
  D. Drop-path:  η=0.0  vs  η=0.5  vs  η=1.0
  E. PCD模式:   skd_only  vs  ckd_only  vs  rcil_full
  F. 下界:      finetune  vs  rcil_full

用法:
  python run.py --ablation_exp no_rc         # 单个实验
  python run.py --ablation_group A           # 运行 A 组 (一对)
  bash ablation/scripts/run_all.sh           # 运行全部
"""

import sys
import argparse

from .ablation_configs import ALL_ABLATIONS, COMPARISON_PAIRS


def get_parser():
    p = argparse.ArgumentParser(description="RCIL Ablation Studies")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--exp", type=str, default=None,
                   help="Single experiment: " + ", ".join(ALL_ABLATIONS.keys()))
    g.add_argument("--group", type=str, default=None,
                   choices=["A", "B", "C", "D", "E", "F", "all"],
                   help="Run a comparison group")
    p.add_argument("--list", action="store_true", help="List all experiments")
    return p


def list_all():
    """列出所有实验及其对照关系"""
    print("\n" + "=" * 65)
    print("  RCIL Ablation: Controlled Comparisons")
    print("=" * 65)

    for pair in COMPARISON_PAIRS:
        print(f"\n  [{pair['name']}] {pair['question']}")
        print(f"    Δ = {pair['delta']}")
        if pair["control"]:
            print(f"    control:  {pair['control']:<20} | {ALL_ABLATIONS[pair['control']]['description']}")
            print(f"    variable: {pair['variable']:<20} | {ALL_ABLATIONS[pair['variable']]['description']}")
        else:
            for exp in pair.get("group", []):
                print(f"    {exp:<20} | {ALL_ABLATIONS[exp]['description']}")

    print("\n" + "=" * 65)
    print(f"  Total: {len(ALL_ABLATIONS)} experiments in 6 comparison groups")
    print("=" * 65 + "\n")


def get_experiments_by_group(group):
    """获取某组的所有实验名"""
    mapping = {
        "A": ["rcil_full", "no_rc"],
        "B": ["rcil_full", "no_pcd"],
        "C": ["rcil_full", "no_frozen"],
        "D": ["droppath_0.0", "droppath_0.5", "droppath_1.0"],
        "E": ["skd_only", "ckd_only", "rcil_full"],
        "F": ["finetune", "rcil_full"],
    }
    if group == "all":
        return list(ALL_ABLATIONS.keys())
    return mapping.get(group, [])


def main():
    args = get_parser().parse_args()

    if args.list:
        list_all()
        return

    if args.exp:
        exps = [args.exp]
    elif args.group:
        exps = get_experiments_by_group(args.group)
    else:
        print("Usage: --exp <name> | --group <A-F> | --list")
        print("Example groups: A=RC, B=PCD, C=Frozen, D=DropPath, E=PCD_Mode, F=LowerBound")
        list_all()
        return

    print(f"Experiments to run: {exps}")
    for exp in exps:
        cfg = ALL_ABLATIONS.get(exp)
        if cfg is None:
            print(f"Unknown: {exp}")
            continue
        print(f"  {exp}: {cfg['description']}")

    print("\n[NOTE] Set --ablation_exp <name> in your run.py command to execute.")
    print("[NOTE] Or use: bash ablation/scripts/run_all.sh")


if __name__ == "__main__":
    main()
