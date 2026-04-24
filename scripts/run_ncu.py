import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import modal
from flashinfer_bench import TraceSet, Solution

app = modal.App("flashinfer-ncu")
trace_volume = modal.Volume.from_name("flashinfer-trace")
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu(solution: Solution, workload_index: int = 0) -> str:
    import os
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    # --- debug: confirm volume is mounted and trace data exists ---
    print(f"[debug] TRACE_SET_PATH = {TRACE_SET_PATH}")
    trace_dir = Path(TRACE_SET_PATH)
    if trace_dir.exists():
        children = list(trace_dir.iterdir())
        print(f"[debug] Contents of {TRACE_SET_PATH}: {[c.name for c in children]}")
    else:
        print(f"[debug] ERROR: {TRACE_SET_PATH} does not exist!")

    # --- load trace set (same pattern as run_modal.py) ---
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    print(f"[debug] Available definitions: {list(trace_set.definitions.keys())}")
    print(f"[debug] Solution definition:   {solution.definition}")

    if solution.definition not in trace_set.definitions:
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set. "
            f"Available: {list(trace_set.definitions.keys())}"
        )

    workloads = trace_set.workloads.get(solution.definition, [])
    print(f"[debug] Found {len(workloads)} workload(s) for this definition")
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    workload = workloads[workload_index]
    print(f"[debug] Using workload [{workload_index}]: {workload.uuid}")

    # --- set FIB_DATASET_PATH so flashinfer_bench_run_ncu can find traces ---
    os.environ["FIB_DATASET_PATH"] = TRACE_SET_PATH
    print(f"[debug] FIB_DATASET_PATH set to {TRACE_SET_PATH}")

    print("[debug] Launching NCU profiler...")
    output = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        set="detailed",
        page="details",
        timeout=300,
    )
    return output


@app.local_entrypoint()
def main():
    from pack_solution import pack_solution  # local-only; not uploaded to Modal container

    print("Packing solution...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning NCU on Modal B200...")
    output = run_ncu.remote(solution)
    print(output)