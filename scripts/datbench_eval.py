import argparse
from uuid import uuid4

import modal

from utils.modal_utils import get_modal_image

results_volume = modal.Volume.from_name("datbench-results", create_if_missing=True)
RESULTS_VOLUME_PATH = "/results"

app = modal.App(name="datbench-eval")
image = get_modal_image()


@app.function(
    image=image,
    volumes={RESULTS_VOLUME_PATH: results_volume},
    timeout=5 * 60,
)
def save_results_to_volume(
    run_id: str,
    results_dicts: list,
    visualizations_data: list,
    config: dict,
) -> str:
    import json
    import os

    run_dir = f"{RESULTS_VOLUME_PATH}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(os.path.join(run_dir, "raw_results.json"), "w") as f:
        json.dump(results_dicts, f, indent=2)

    if visualizations_data:
        with open(os.path.join(run_dir, "visualizations_data.json"), "w") as f:
            json.dump(visualizations_data, f)

    results_volume.commit()

    print(f"Results saved to volume at: {run_dir}")
    return run_dir


@app.function(
    image=image,
    gpu="A100",
    timeout=30 * 60,
    retries=1,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_pruner_subset(
    pruner_name: str,
    pruner_params: dict,
    subset: str,
    sample_indices: list,
    model_name: str,
    max_new_tokens: int,
) -> list:
    """
    Evaluate a single pruner on a single subset.

    This function runs on a separate GPU container for each (pruner, subset) combination,
    enabling parallel evaluation across Modal's infrastructure.

    Returns:
        Dict with results and visualizations
    """
    import transformers
    from datasets import load_dataset
    from datbench import DatBenchEvaluator, VLMResponse

    from eval.config import get_pruner
    from eval.results import SampleResult
    from model import load_model, reset_inference_context, run_inference

    transformers.logging.set_verbosity_error()

    print(
        f"Starting evaluation: {pruner_name} on {subset} ({len(sample_indices)} samples)"
    )

    pruner = get_pruner(pruner_name, **pruner_params)

    print(f"Loading model: {model_name}")
    model, processor = load_model(
        model_name=model_name, pruner=pruner, rope_config=None
    )

    print(f"Loading DatBench subset: {subset}")
    dataset = load_dataset("DatologyAI/DatBench", subset, split="test")

    if sample_indices:
        valid_indices = [i for i in sample_indices if i < len(dataset)]
        dataset = dataset.select(valid_indices)
    else:
        valid_indices = None

    evaluator = DatBenchEvaluator(dataset, subset)
    tasks = evaluator.get_inference_tasks()

    results = []
    visualizations_data = []
    vlm_responses = []
    task_id_to_idx = {}

    for idx, task in enumerate(tasks):
        sample_id = valid_indices[idx] if valid_indices else idx
        print(f"  Processing sample {sample_id} (task_id={task.id})...")

        reset_inference_context(model)

        try:
            result = run_inference(
                model,
                processor,
                task.image,
                task.question,
                max_new_tokens=max_new_tokens,
                timed=True,
            )

            prediction = result.decode_output(processor)
            surviving_indices = result.get_surviving_indices()

            vlm_responses.append(VLMResponse(id=task.id, raw_output=prediction))
            task_id_to_idx[task.id] = len(results)

            if surviving_indices:
                spatial_merge_size = model.model.visual.config.spatial_merge_size
                patch_size = model.model.visual.config.patch_size
                visualizations_data.append(
                    {
                        "sample_id": sample_id,
                        "surviving_indices": [
                            i.cpu().tolist() if hasattr(i, "cpu") else list(i)
                            for i in surviving_indices
                        ],
                        "image_grid_thw": result.image_grid_thw.cpu().tolist(),
                        "spatial_merge_size": spatial_merge_size,
                        "patch_size": patch_size,
                    }
                )

            sample_result = SampleResult(
                sample_id=sample_id,
                subset=subset,
                pruner_name=pruner_name,
                success=True,
                question=task.question,
                prediction=prediction,
                ground_truth=evaluator.samples_by_id[task.id].answer,
                ttft_ms=result.timing.ttft_ms if result.timing else None,
                decode_latency_ms=result.timing.decode_latency_ms
                if result.timing
                else None,
                num_output_tokens=result.timing.num_tokens if result.timing else None,
                initial_visual_tokens=len(surviving_indices[0])
                if surviving_indices
                else 0,
                final_visual_tokens=len(surviving_indices[-1])
                if surviving_indices
                else 0,
                pruning_ratio=result.get_pruning_ratio(),
                pruning_steps=max(0, len(surviving_indices) - 1),
            )

        except Exception as e:
            print(f"    Error on sample {sample_id}: {e}")
            sample_result = SampleResult(
                sample_id=sample_id,
                subset=subset,
                pruner_name=pruner_name,
                success=False,
                question=task.question,
                ground_truth=evaluator.samples_by_id[task.id].answer,
                error=str(e),
            )

        results.append(sample_result)

    print("Computing scores with DatBench evaluator...")
    report = evaluator.compute_metrics(vlm_responses)
    for r in report.results:
        results[task_id_to_idx[r.id]].score = r.score

    results_dicts = [r.to_dict() for r in results]

    print(f"Completed {pruner_name} on {subset}: {len(results)} samples processed")

    return {
        "results": results_dicts,
        "visualizations": visualizations_data,
        "report_summary": report.summary
        if report and hasattr(report, "summary")
        else {},
    }


@app.local_entrypoint()
def main(
    sample_count: int = 100,
    seed: int = 42,
    pruner: str = "baseline",
    subsets: str = "math",
    max_new_tokens: int = 1024,
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
):
    import random

    from eval.config import DATBENCH_SUBSETS, DEFAULT_PARAMS
    from eval.results import SampleResult

    subset_names = [s.strip() for s in subsets.split(",") if s.strip()]

    if pruner not in DEFAULT_PARAMS:
        print(f"Unknown pruner: {pruner}")
        print(f"Available: {list(DEFAULT_PARAMS.keys())}")
        return

    for s in subset_names:
        if s not in DATBENCH_SUBSETS:
            print(f"Unknown subset: {s}")
            print(f"Available: {DATBENCH_SUBSETS}")
            return

    run_id = str(uuid4())

    if not subset_names:
        subset_names = DATBENCH_SUBSETS

    print("DatBench Evaluation")
    print("==================")
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"Pruner: {pruner}")
    print(f"Subsets: {subset_names}")
    print(f"Samples per subset: {sample_count}")
    print()

    random.seed(seed)
    sample_indices = list(range(sample_count))

    config = {
        "run_id": run_id,
        "model_name": model_name,
        "pruner": pruner,
        "subsets": subset_names,
        "sample_count": sample_count,
        "sample_indices": sample_indices,
        "random_seed": seed,
        "max_new_tokens": max_new_tokens,
    }

    print(f"Spawning {len(subset_names)} evaluation jobs...")
    function_calls = [
        evaluate_pruner_subset.spawn(
            pruner_name=pruner,
            pruner_params={},
            subset=subset,
            sample_indices=sample_indices,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        for subset in subset_names
    ]

    all_results = []
    all_visualizations = []

    for output in modal.FunctionCall.gather(*function_calls):
        results = output["results"]
        subset = results[0]["subset"] if results else "unknown"
        success_count = sum(1 for r in results if r["success"])
        print(f"  {subset}: {success_count}/{len(results)} successful")

        for r in results:
            all_results.append(SampleResult.from_dict(r))
        all_visualizations.extend(output.get("visualizations", []))

    print(f"\nTotal results collected: {len(all_results)}")

    print("\nSaving results to Modal volume...")
    results_dicts = [r.to_dict() for r in all_results]
    volume_path = save_results_to_volume.remote(
        run_id=run_id,
        results_dicts=results_dicts,
        visualizations_data=all_visualizations,
        config=config,
    )

    success_count = sum(1 for r in all_results if r.success)
    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE")
    print("=" * 55)
    print(f"Results: {success_count}/{len(all_results)} successful")
    print(f"Saved to: {volume_path}")
    print(f"\nDownload: modal volume get datbench-results {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DatBench Evaluation for Visual Token Pruning"
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pruner", type=str, default="baseline")
    parser.add_argument("--subsets", type=str, default="math")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")

    args = parser.parse_args()
    main(**vars(args))
