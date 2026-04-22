import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import modal
from flashinfer_bench import TraceSet, Solution, Workload
from flashinfer_bench.agents import flashinfer_bench_run_ncu

app = modal.App("flashinfer-ncu")
trace_volume = modal.Volume.from_name("flashinfer-trace")
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, volumes={TRACE_SET_PATH: trace_volume})
def fetch_workload(definition: str) -> Workload:
    """Load workload from Modal volume and return it to the local caller."""
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{definition}'")
    return workloads[0]


@app.local_entrypoint()
def main():
    from pack_solution import pack_solution  # local-only; not uploaded to Modal container

    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    # Fetch workload object from the volume, then run NCU locally as an agent call
    workload = fetch_workload.remote(solution.definition)

    output = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        set="detailed",
        page="details",
        timeout=300,
    )
    print(output)