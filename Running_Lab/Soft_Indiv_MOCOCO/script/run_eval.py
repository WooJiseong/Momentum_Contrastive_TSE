from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


LAB_DIR = Path(__file__).resolve().parents[1]
PN_MOCOCO_DIR = LAB_DIR.parent / "PN_MOCOCO"
sys.path.insert(0, str(PN_MOCOCO_DIR))
sys.path.insert(0, str(LAB_DIR))


def merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    override = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load((LAB_DIR / override["base_config"]).resolve().read_text(encoding="utf-8"))
    merge(base, override)
    for section, keys in {"paths": ("initial_model_ckpt", "encoder_override_ckpt")}.items():
        for key in keys:
            value = base.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                base[section][key] = str((LAB_DIR / value).resolve())
    base.setdefault("eval", {})["checkpoint"] = str(Path(args.checkpoint).resolve())
    base["eval"]["out_json"] = str(Path(args.out).resolve())
    runtime = Path(args.out).resolve().parent / "config_eval_resolved.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    sys.argv = [
        "eval_tfgridnet.py", "--config", str(runtime), "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--out", str(Path(args.out).resolve()),
    ]
    import eval_tfgridnet

    eval_tfgridnet.main()


if __name__ == "__main__":
    main()
