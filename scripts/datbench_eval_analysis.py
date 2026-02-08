import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def download_results(run_id: str, output_dir: Path) -> Path:
    """Download results from Modal volume."""
    run_dir = output_dir / run_id
    if run_dir.exists():
        print(f"Results already exist at {run_dir}, skipping download")
        return run_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "datbench-results", run_id, str(output_dir)],
        check=True,
    )
    return run_dir


def report_metrics(results_dir: Path):
    """Report metrics from the run."""
    with open(results_dir / "config.json") as f:
        config = json.load(f)

    with open(results_dir / "raw_results.json") as f:
        results = json.load(f)

    print("\n" + "=" * 60)
    print("RUN CONFIGURATION")
    print("=" * 60)
    print(f"Run ID: {config.get('run_id', 'N/A')}")
    print(f"Model: {config.get('model_name', 'N/A')}")
    print(f"Pruner: {config.get('pruner', 'N/A')}")
    print(f"Subsets: {config.get('subsets', [])}")
    print(f"Samples per subset: {config.get('sample_count', 'N/A')}")

    by_subset = defaultdict(list)
    for r in results:
        by_subset[r["subset"]].append(r)

    print("\n" + "=" * 60)
    print("RESULTS BY SUBSET")
    print("=" * 60)

    for subset, subset_results in sorted(by_subset.items()):
        successful = [r for r in subset_results if r.get("success")]
        scores = [r["score"] for r in successful if r.get("score") is not None]
        ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
        pruning_ratios = [
            r["pruning_ratio"] for r in successful if r.get("pruning_ratio") is not None
        ]

        print(f"\n{subset}:")
        print(f"  Success: {len(successful)}/{len(subset_results)}")
        if scores:
            print(f"  Avg Score: {sum(scores) / len(scores):.3f}")
        if ttfts:
            print(f"  Avg TTFT: {sum(ttfts) / len(ttfts):.1f} ms")
        if pruning_ratios:
            print(
                f"  Avg Pruning Ratio: {sum(pruning_ratios) / len(pruning_ratios):.2%}"
            )

    all_successful = [r for r in results if r.get("success")]
    all_scores = [r["score"] for r in all_successful if r.get("score") is not None]

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"Total: {len(all_successful)}/{len(results)} successful")
    if all_scores:
        print(f"Avg Score: {sum(all_scores) / len(all_scores):.3f}")


def save_pruning_progression(
    image: Image.Image,
    surviving_indices: list,
    image_grid_thw: list,
    spatial_merge_size: int,
    patch_size: int,
    output_dir: Path,
):
    """Save step-by-step visualization of pruning progression."""
    os.makedirs(output_dir, exist_ok=True)
    image_np = np.array(image)

    H_patches, W_patches = int(image_grid_thw[1]), int(image_grid_thw[2])
    H_tok = H_patches // spatial_merge_size
    W_tok = W_patches // spatial_merge_size
    token_size = spatial_merge_size * patch_size
    total_tokens = H_tok * W_tok

    # Save original
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np.astype(np.uint8))
    plt.axis("off")
    plt.savefig(output_dir / "step_0_original.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    for step_idx, step_indices in enumerate(surviving_indices[1:], start=1):
        copy_image_np = image_np.copy().astype(float) * 0.15

        for idx in step_indices:
            idx = int(idx)
            row, col = idx // W_tok, idx % W_tok
            pr, pc = row * token_size, col * token_size
            pr_end = min(pr + token_size, image_np.shape[0])
            pc_end = min(pc + token_size, image_np.shape[1])
            copy_image_np[pr:pr_end, pc:pc_end] = image_np[pr:pr_end, pc:pc_end]

        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(copy_image_np.astype(np.uint8))
        ax.set_xticks(np.arange(0, image_np.shape[1], token_size))
        ax.set_yticks(np.arange(0, image_np.shape[0], token_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="gray", alpha=0.3, linewidth=0.5)
        ax.tick_params(length=0)

        surviving_count = len(step_indices)
        pruned_ratio = (
            1.0 - (surviving_count / total_tokens) if total_tokens > 0 else 0.0
        )
        ax.set_title(
            f"Step {step_idx}: {surviving_count}/{total_tokens} tokens ({pruned_ratio:.1%} pruned)"
        )

        plt.savefig(output_dir / f"step_{step_idx}.png", bbox_inches="tight", dpi=150)
        plt.close()


def generate_visualizations(results_dir: Path, output_dir: Path):
    """Generate visualizations for each subset."""
    from datasets import load_dataset

    with open(results_dir / "config.json") as f:
        config = json.load(f)

    viz_data_path = results_dir / "visualizations_data.json"
    if not viz_data_path.exists():
        print("No visualization data found")
        return

    with open(viz_data_path) as f:
        viz_data = json.load(f)

    if not viz_data:
        print("Visualization data is empty")
        return

    with open(results_dir / "raw_results.json") as f:
        results = json.load(f)

    sample_to_subset = {r["sample_id"]: r["subset"] for r in results}

    by_subset = defaultdict(list)
    for v in viz_data:
        subset = sample_to_subset.get(v["sample_id"])
        if subset:
            by_subset[subset].append(v)

    for subset in config["subsets"]:
        subset_viz = by_subset.get(subset, [])
        if not subset_viz:
            print(f"No visualization data for {subset}")
            continue

        v = subset_viz[0]
        sample_id = v["sample_id"]

        print(f"\nGenerating visualization for {subset} (sample {sample_id})...")
        dataset = load_dataset("DatologyAI/DatBench", subset, split="test")

        if sample_id >= len(dataset):
            print(f"  Sample {sample_id} out of range")
            continue

        image = dataset[sample_id]["image"]
        save_pruning_progression(
            image=image,
            surviving_indices=v["surviving_indices"],
            image_grid_thw=v["image_grid_thw"],
            spatial_merge_size=v["spatial_merge_size"],
            patch_size=v["patch_size"],
            output_dir=output_dir / subset,
        )

        print(f"  Saved to {output_dir / subset}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze DatBench evaluation results")
    parser.add_argument("run_id", help="Run ID to analyze")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./assets/datbench_eval"),
        help="Output directory",
    )

    args = parser.parse_args()

    results_dir = download_results(args.run_id, args.output_dir)
    report_metrics(results_dir)

    if args.visualize:
        generate_visualizations(
            results_dir, args.output_dir / args.run_id / "visualizations"
        )


if __name__ == "__main__":
    main()
