"""
Combine per-chain partial_<tag>.json files (written by
run_bayesian_inversion_ood.py --single ...) into a single summary.json,
same schema as a sequential run_bayesian_inversion_ood.py run would produce.

Usage:
    python merge_summaries.py Results/mcmc_n_samples_.../
"""
import glob
import json
import os
import sys

ORDER = [
    "true_model",
    "DeepONet", "DeepONet-corrected",
    "PCANet", "PCANet-corrected",
    "FNO", "FNO-corrected",
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <savepath_base>")
    savepath_base = sys.argv[1]

    partials = {}
    for path in glob.glob(os.path.join(savepath_base, "partial_*.json")):
        with open(path) as f:
            result = json.load(f)
        partials[result["tag"]] = result

    missing = [tag for tag in ORDER if tag not in partials]
    if missing:
        print(f"Warning: missing partials for {missing}; merging what's available.")

    summary = [partials[tag] for tag in ORDER if tag in partials]

    summary_path = os.path.join(savepath_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved merged summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
