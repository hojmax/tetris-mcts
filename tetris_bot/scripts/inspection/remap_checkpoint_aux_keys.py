"""Remap old aux_fc/aux_ln checkpoint keys to aux_mlp.* naming."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from simple_parsing import ArgumentParser

KEY_MAP = {
    "aux_fc.weight": "aux_mlp.0.weight",
    "aux_fc.bias": "aux_mlp.0.bias",
    "aux_ln.weight": "aux_mlp.1.weight",
    "aux_ln.bias": "aux_mlp.1.bias",
}


@dataclass
class Args:
    checkpoint: Path
    output: Path | None = None
    backup: bool = True


def main(args: Args) -> None:
    output = args.output or args.checkpoint
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    dicts_to_remap = ["model_state_dict"]
    if "ema_state_dict" in state:
        dicts_to_remap.append("ema_state_dict")

    remapped = 0
    for dict_name in dicts_to_remap:
        sd = state[dict_name]
        for old_key, new_key in KEY_MAP.items():
            if old_key in sd:
                sd[new_key] = sd.pop(old_key)
                remapped += 1

    if remapped == 0:
        print("No keys needed remapping — checkpoint already uses aux_mlp.* naming.")
        return

    if args.backup and output == args.checkpoint:
        backup_path = args.checkpoint.with_suffix(".pt.bak")
        shutil.copy2(args.checkpoint, backup_path)
        print(f"Backup saved to {backup_path}")

    torch.save(state, output)
    print(f"Remapped {remapped} keys, saved to {output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)
