"""Merge a completed adapter on CPU for an external VLMEvalKit evaluation."""
import argparse
from pathlib import Path

from synselect.utils.config import load_config, local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--adapter", help="Optional checkpoint directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    adapter = local_path(args.adapter or cfg["output_dir"])
    target = local_path(args.output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Export needs an empty output directory: {target}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(adapter)
    from synselect.runtime import build_model, build_processor
    model = build_model(cfg, str(adapter), trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(str(target), safe_serialization=True)
    build_processor(cfg).save_pretrained(str(target))
    print(f"Merged Hugging Face model: {Path(target).resolve()}")


if __name__ == "__main__":
    main()
