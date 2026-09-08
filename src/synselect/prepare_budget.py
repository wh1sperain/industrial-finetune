"""Preflight all sampled datasets before spending the SFT budget on the server."""
import argparse
import hashlib
import json
from pathlib import Path

from synselect.data import load_rows
from synselect.utils.config import load_config, local_path


def prepare(sft_name, grpo_name, output):
    from PIL import Image
    sft, grpo = load_config(sft_name), load_config(grpo_name)
    for key in ("model", "prompt", "processor", "eval"):
        if sft[key] != grpo[key]:
            raise ValueError(f"Budget stage settings must match: {key}")
    if local_path(grpo["sft_adapter"]) != Path(sft["output_dir"]):
        raise ValueError("GRPO does not point to this pipeline's SFT output")
    train_path = local_path(grpo["dataset"]["annotations"]).resolve()
    eval_path = local_path(grpo["eval"]["dataset"]["annotations"]).resolve()
    if train_path == eval_path:
        raise ValueError("Training and validation annotations must be independent")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collections = {}
    for name, cfg, spec in (("sft", sft, sft["dataset"]),
                            ("grpo", grpo, grpo["dataset"]),
                            ("eval", grpo, grpo["eval"]["dataset"])):
        rows, report = load_rows(cfg, spec, return_selection=True)
        digest = hashlib.sha256()
        with local_path(spec["annotations"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        report = dict(report or {}, annotations_sha256=digest.hexdigest())
        images = {str(Path(r["image"]).resolve()) for r in rows}
        for path in images:
            with Image.open(path) as image:
                image.verify()
        collections[name] = images
        (output / f"{name}_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{name}: {len(rows)} questions, {len(images)} images verified", flush=True)
    if collections["eval"] & (collections["sft"] | collections["grpo"]):
        raise ValueError("Selected training/validation images overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-experiment", required=True)
    parser.add_argument("--grpo-experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.sft_experiment, args.grpo_experiment, args.output)


if __name__ == "__main__":
    main()
