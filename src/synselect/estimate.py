"""Print assumptions-based estimates without GPU, dataset downloads or model loading."""
import argparse
import json

from synselect.utils.config import load_config
from synselect.utils.timing import planning_estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--samples", type=int, help="Actual number of training questions")
    args = parser.parse_args()
    if args.samples is not None and args.samples < 1:
        parser.error("samples must be positive")
    cfg = load_config(args.experiment)
    counts = [args.samples] if args.samples else cfg["estimate"]["sample_counts"]
    print(json.dumps([planning_estimate(cfg, n) for n in counts], indent=2))


if __name__ == "__main__":
    main()
