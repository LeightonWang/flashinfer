import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import modal
from flashinfer_bench import TraceSet, Solution
from flashinfer_bench.agents import flashinfer_bench_run_ncu
from pack_solution import pack_solution

app = modal.App("flashinfer-ncu")
trace_volume = modal.Volume.from_name("flashinfer-trace")
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu_remote(solution: Solution):
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workload = trace_set.workloads[solution.definition][0]  # 取第一个 workload
    
    output = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        set="detailed",
        page="details",
        timeout=300,
    )
    print(output)
    return output

@app.local_entrypoint()
def main():
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    run_ncu_remote.remote(solution)