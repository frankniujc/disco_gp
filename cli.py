#!/usr/bin/env python3
"""
run_cli.py — DiscoGP CLI with task-specific default hparams + light presets

- Main defaults match your latest run.py (preset='main').
- All weight/edge hparams are available as CLI flags and override defaults.
- Different tasks can have different default bundles via PRESETS[task][preset].

Examples:
  # main defaults (BLiMP + GPT-2 + your hparams)
  python run_cli.py

  # PARArel with preset2 (fill numbers in PRESETS), and relation/data specified
  python run_cli.py --task pararel --preset preset2 \
    --pararel-rels "P36" --pararel-data ./data/pararel_data_all.json

  # IOI with preset1 and Qwen 1.7B
  python run_cli.py --task ioi --preset preset1 --model Qwen/Qwen3-1.7B
"""

import argparse
import gc
import torch

from disco_gp import DiscoGPTransformer, Config, set_seed

# ----------------------------- Task Presets -----------------------------
# Edit these numbers per task/preset as you like. 'main' mirrors your current run.py.
# NOTE: CLI flags override these. If a flag is omitted, we fall back to the chosen preset.

DEFAULT = {
    "weight": {
        "use_weight_masks": True,
        "gs_temp_weight": 0.01,
        "logits_w_init": 1.0,
        "lr": 1.0,
        "lambda_sparse_init": 1.0,
        "lambda_complete_init": 1.0,
        "min_times_lambda_sparse": 1.0,
        "max_times_lambda_sparse": 1000.0,
        "train_epochs": 500,
        "n_epoch_warmup_lambda_sparse": 500,
        "n_epoch_cooldown_lambda_sparse": 1,
    },
    "edge": {
        "use_edge_masks": True,
        "gs_temp_edge": 1.0,
        "logits_e_init": 1.0,
        "lr": 0.1,
        "lambda_sparse_init": 1.0,
        "lambda_complete_init": 1.0,
        "min_times_lambda_sparse": 0.01,
        "max_times_lambda_sparse": 100.0,
        "train_epochs": 100,
        "n_epoch_warmup_lambda_sparse": 20,
        "n_epoch_cooldown_lambda_sparse": 20,
    },
}

def _merge(a: dict, b: dict) -> dict:
    """Shallow merge: values in b override a (used for preset + overrides)."""
    out = dict(a)
    out.update({k: v for k, v in b.items() if v is not None})
    return out

PRESETS = {
    "pararel": {
        "we": _merge(
            DEFAULT, {
                'lr': 0.1,
            }),
        'w': _merge(
            DEFAULT, {
                'lr': 0.1,
            }),
        'e': _merge(
            DEFAULT, {
                'lr': 0.1,
            }
        ),
    },
    "ioi": {
        "we": DEFAULT.copy(),
        'w': DEFAULT.copy(),
        'e': DEFAULT.copy(),
    },
    "blimp": {
        'we': DEFAULT.copy(),
        'w': DEFAULT.copy(),
        'e': DEFAULT.copy(),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiscoGP CLI")

    # Task + modes
    p.add_argument("--task", choices=["pararel", "ioi", "blimp"])
    p.add_argument("--modes", type=str, default="we", help="'w' (weights), 'e' (edges), 'we' (both)")
    p.add_argument("--seed", type=int, default=42)

    # Task-specific knobs
    p.add_argument("--pararel-rels", type=str, default="P36")
    p.add_argument("--pararel-data", type=str, default="./data/pararel_data_all.json")
    p.add_argument("--n-ioi-data", type=int, default=1000)
    p.add_argument("--blimp-paradigm", type=str, default="anaphor_number_agreement")

    # Data / splits
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--dev-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)

    # Model / dtype
    p.add_argument("--model", type=str, default="gpt2",
                   help="gpt2 | Qwen/Qwen3-0.6B | Qwen/Qwen3-1.7B | meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    # W&B
    p.add_argument("--use-wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default="DiscoGP Release")
    p.add_argument("--wandb-entity", type=str, default="WANDB_ENTITY")

    # Experiment meta
    p.add_argument("--evaluate-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--exp-name", type=str, default="quickstart")

    # ---------------- Weight pruning hparams (flags) ----------------
    # Use None defaults so we can detect "user did not pass" and fill from preset
    p.add_argument("--w-gs-temp", type=float, default=None)
    p.add_argument("--w-logits-init", type=float, default=None)
    p.add_argument("--w-lr", type=float, default=None)
    p.add_argument("--w-lambda-sparse", type=float, default=None)
    p.add_argument("--w-lambda-complete", type=float, default=None)
    p.add_argument("--w-min-times", type=float, default=None)
    p.add_argument("--w-max-times", type=float, default=None)
    p.add_argument("--w-train-epochs", type=int, default=None)
    p.add_argument("--w-warmup", type=int, default=None)
    p.add_argument("--w-cooldown", type=int, default=None)
    # tri-state bool for w-use
    p.add_argument("--w-use", dest="w_use", action="store_true")
    p.add_argument("--no-w-use", dest="w_use", action="store_false")
    p.set_defaults(w_use=None)

    # ----------------- Edge pruning hparams (flags) -----------------
    p.add_argument("--e-gs-temp", type=float, default=None)
    p.add_argument("--e-logits-init", type=float, default=None)
    p.add_argument("--e-lr", type=float, default=None)
    p.add_argument("--e-lambda-sparse", type=float, default=None)
    p.add_argument("--e-lambda-complete", type=float, default=None)
    p.add_argument("--e-min-times", type=float, default=None)
    p.add_argument("--e-max-times", type=float, default=None)
    p.add_argument("--e-train-epochs", type=int, default=None)
    p.add_argument("--e-warmup", type=int, default=None)
    p.add_argument("--e-cooldown", type=int, default=None)
    # tri-state bool for e-use
    p.add_argument("--e-use", dest="e_use", action="store_true")
    p.add_argument("--no-e-use", dest="e_use", action="store_false")
    p.set_defaults(e_use=None)

    return p.parse_args()

def build_hparams(args: argparse.Namespace):
    # 1) Start from the selected task + preset defaults
    task = args.task
    modes = args.modes
    base = PRESETS.get(task, {}).get(modes, {})
    base_w = dict(base.get("weight", {}))
    base_e = dict(base.get("edge", {}))

    # 2) Prepare CLI overrides (only include keys with non-None values)
    cli_w = {
        "use_weight_masks": args.w_use,
        "gs_temp_weight": args.w_gs_temp,
        "logits_w_init": args.w_logits_init,
        "lr": args.w_lr,
        "lambda_sparse_init": args.w_lambda_sparse,
        "lambda_complete_init": args.w_lambda_complete,
        "min_times_lambda_sparse": args.w_min_times,
        "max_times_lambda_sparse": args.w_max_times,
        "train_epochs": args.w_train_epochs,
        "n_epoch_warmup_lambda_sparse": args.w_warmup,
        "n_epoch_cooldown_lambda_sparse": args.w_cooldown,
    }
    cli_e = {
        "use_edge_masks": args.e_use,
        "gs_temp_edge": args.e_gs_temp,
        "logits_e_init": args.e_logits_init,
        "lr": args.e_lr,
        "lambda_sparse_init": args.e_lambda_sparse,
        "lambda_complete_init": args.e_lambda_complete,
        "min_times_lambda_sparse": args.e_min_times,
        "max_times_lambda_sparse": args.e_max_times,
        "train_epochs": args.e_train_epochs,
        "n_epoch_warmup_lambda_sparse": args.e_warmup,
        "n_epoch_cooldown_lambda_sparse": args.e_cooldown,
    }

    # 3) Merge: preset → CLI overrides
    merged_w = _merge(base_w, cli_w)
    merged_e = _merge(base_e, cli_e)

    # 5) Wrap into Configs
    weight_hparams = Config(**merged_w)
    edge_hparams = Config(**merged_e)
    return weight_hparams, edge_hparams

def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Dtype
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # 0) W&B
    wandb_cfg = Config(
        use_wandb=args.use_wandb,
        wandb_project_name=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # 1) Weight/edge hparams (task + preset defaults, then CLI overrides)
    weight_hparams, edge_hparams = build_hparams(args)

    # 2) Task config
    if args.task == "pararel":
        task_cfg = Config(
            task_type="pararel",
            pararel_rel_ids=args.pararel_rels,
            pararel_data_path=args.pararel_data,
            batch_size=args.batch_size,
            ds_split_ratios=(args.train_split, args.dev_split, args.test_split),
        )
    elif args.task == "ioi":
        task_cfg = Config(
            task_type="ioi",
            n_ioi_data=args.n_ioi_data,
            batch_size=args.batch_size,
            ds_split_ratios=(args.train_split, args.dev_split, args.test_split),
        )
    else:  # blimp
        task_cfg = Config(
            task_type="blimp",
            paradigm=args.blimp_paradigm,
            batch_size=args.batch_size,
            ds_split_ratios=(args.train_split, args.dev_split, args.test_split),
        )

    # 3) Experiment meta
    exp_cfg = Config(
        evaluate_every=args.evaluate_every,
        save_every=args.save_every,
        output_dir_path=args.output_dir,
        exp_name=args.exp_name,
    )

    # 4) Model
    model_cfg = Config.from_tl(args.model, dtype=dtype)

    # 5) Merge + run
    cfg = Config.from_configs(
        wandb=wandb_cfg,
        weight=weight_hparams,
        edge=edge_hparams,
        task=task_cfg,
        model=model_cfg,
        exp=exp_cfg,
    )

    print(f"[Step] Using task='{args.task}', modes='{args.modes}'")
    print("[Step] Loading model + data…")
    model = DiscoGPTransformer.from_pretrained(cfg)

    print("[Step] Setup the experiment…")
    model.setup_experiment()

    print("[Step] Baseline evaluation:")
    model.evaluate_and_report(epoch=0, mode="baseline")

    print(f"[Step] Pruning (modes='{args.modes}')…")
    model.search(modes=args.modes)

    print("[Step] Final evaluation:")
    model.evaluate_and_report(epoch="final", mode="pruned")

    print("[Step] Teardown…")
    model.teardown_experiment()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
