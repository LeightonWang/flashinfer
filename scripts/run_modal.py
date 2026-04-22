"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def _run_with_optional_profile(benchmark: Benchmark, dump_traces: bool, enable_profile: bool, profile_output: str):
    """Run benchmark with optional torch profiler trace export."""
    if not enable_profile:
        return benchmark.run_all(dump_traces=dump_traces)

    import os
    import torch
    from pathlib import Path

    out_path = Path(profile_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase_output = out_path.with_suffix(".phase.json")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    prev_phase_mark = os.environ.get("FIB_PROFILE_PHASES")
    prev_phase_output = os.environ.get("FIB_PHASE_TIMING_OUTPUT")
    os.environ["FIB_PROFILE_PHASES"] = "1"
    os.environ["FIB_PHASE_TIMING_OUTPUT"] = str(phase_output)
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            result_trace_set = benchmark.run_all(dump_traces=dump_traces)
    finally:
        if prev_phase_mark is None:
            os.environ.pop("FIB_PROFILE_PHASES", None)
        else:
            os.environ["FIB_PROFILE_PHASES"] = prev_phase_mark

        if prev_phase_output is None:
            os.environ.pop("FIB_PHASE_TIMING_OUTPUT", None)
        else:
            os.environ["FIB_PHASE_TIMING_OUTPUT"] = prev_phase_output

    prof.export_chrome_trace(str(out_path))

    summary_output = out_path.with_suffix(".summary.txt")
    summary_table = prof.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        row_limit=60,
    )
    with open(summary_output, "w", encoding="utf-8") as f:
        f.write(summary_table)

    print(f"\n[profile] Chrome trace written to: {out_path}")
    print(f"[profile] Kernel summary written to: {summary_output}")
    if phase_output.exists():
        print(f"[profile] Phase timing written to: {phase_output}")
    else:
        print(f"[profile] Phase timing file not found at: {phase_output}")
    print("\n[profile] Top events (sorted by self CUDA/CPU time):")
    print(summary_table)
    return result_trace_set


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = 0,
    enable_profile: bool = False,
    profile_output: str = "/tmp/fib_profile_modal_trace.json",
) -> dict:
    """Run benchmark on Modal B200 and return results.

    Args:
        max_workloads: Max number of workloads to run. 0 means all.
    """
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    import os
    from pathlib import Path

    prev_phase_mark = os.environ.get("FIB_PROFILE_PHASES")
    prev_phase_output = os.environ.get("FIB_PHASE_TIMING_OUTPUT")
    if enable_profile:
        os.environ["FIB_PROFILE_PHASES"] = "1"
        os.environ["FIB_PHASE_TIMING_OUTPUT"] = str(Path(profile_output).with_suffix(".phase.json"))

    try:
        trace_set = TraceSet.from_path(TRACE_SET_PATH)

        if solution.definition not in trace_set.definitions:
            raise ValueError(f"Definition '{solution.definition}' not found in trace set")

        definition = trace_set.definitions[solution.definition]
        workloads = trace_set.workloads.get(solution.definition, [])

        if not workloads:
            raise ValueError(f"No workloads found for definition '{solution.definition}'")

        if max_workloads > 0:
            workloads = workloads[:max_workloads]
            print(f"Running {len(workloads)} of {len(trace_set.workloads.get(solution.definition, []))} workloads")

        bench_trace_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: workloads},
            traces={definition.name: []},
        )

        benchmark = Benchmark(bench_trace_set, config)
        result_trace_set = _run_with_optional_profile(
            benchmark,
            dump_traces=True,
            enable_profile=enable_profile,
            profile_output=profile_output,
        )

        traces = result_trace_set.traces.get(definition.name, [])
        results = {definition.name: {}}

        for trace in traces:
            if trace.evaluation:
                entry = {
                    "status": trace.evaluation.status.value,
                    "solution": trace.solution,
                    "log": trace.evaluation.log,
                }
                if trace.evaluation.performance:
                    entry["latency_ms"] = trace.evaluation.performance.latency_ms
                    entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                    entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
                if trace.evaluation.correctness:
                    entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                    entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
                results[definition.name][trace.workload.uuid] = entry

        return results
    finally:
        if prev_phase_mark is None:
            os.environ.pop("FIB_PROFILE_PHASES", None)
        else:
            os.environ["FIB_PROFILE_PHASES"] = prev_phase_mark

        if prev_phase_output is None:
            os.environ.pop("FIB_PHASE_TIMING_OUTPUT", None)
        else:
            os.environ["FIB_PHASE_TIMING_OUTPUT"] = prev_phase_output


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            if result.get("log"):
                print(f"    Log: {result['log']}")


@app.local_entrypoint()
def main(max_workloads: int = 0, profile: bool = False, profile_output: str = "/tmp/fib_profile_modal_trace.json"):
    """Pack solution and run benchmark on Modal.

    Args:
        max_workloads: Max number of workloads to run. 0 means all.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(
        solution,
        max_workloads=max_workloads,
        enable_profile=profile,
        profile_output=profile_output,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)