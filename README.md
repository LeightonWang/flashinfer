# [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

> Some of the original content of this README is not applicable. So I made some edits to make it work.

[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) is our official framework to evaluate your AI-generated kernels.

<!-- ## Competition Tracks

The competition features three tracks, each targeting a critical LLM operation:

| Track | Description |
|-------|-------------|
| **fused_moe** | Fused Mixture-of-Experts kernel for efficient expert routing and computation |
| **sparse_attention** | Sparse attention mechanisms for long-context inference |
| **gated_delta_net** | Gated delta network operations for efficient state updates |

**Fork this template once per track** you want to compete in (separate repos for each track). -->

## Getting Started

### 1. Fork This Template

Click "Use this template" or fork this repository to create your solution repo.

### 2. Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git@main modal
```
> Edited: Use source from github instead of PyPI since the latest version on PyPI is 0.1.0 which lacks `flashinfer_moe.agents` module.

### 3. Download the TraceSet

We provide kernel definitions and workloads in [FlashInfer-Trace format](https://bench.flashinfer.ai/docs/flashinfer-trace). Clone the competition dataset from HuggingFace:

```bash
# You might need to install git-lfs first
apt-get update && apt-get install -y git-lfs
# Install
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

Set the environment variable:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
```

### 4. Configure Your Solution

Edit `config.toml` to set your track and team info:

```toml
[solution]
name = "my-team-solution-v1"      # Solution name
definition = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"          # Track: fused_moe | sparse_attention | gated_delta_net
author = "team-name"              # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "kernel.py::run"            # Kernel function name
```

### 5. Implement Your Kernel

**For Triton:**
Edit `solution/triton/kernel.py` with your implementation.

**For CUDA:**
Edit `solution/cuda/kernel.cu` and `solution/cuda/binding.py` with your implementation.

## Development Workflow

### Pack Your Solution

Generate `solution.json` from your source files:

```bash
python scripts/pack_solution.py
```

### Run Local Benchmarks

Test your solution on your local GPU:

```bash
python scripts/run_local.py
```

Options:
- `--max-workloads`: Run on a subset of workloads for quick testing (default: all workloads)

For example, run only the first 3 workloads:

```bash
python scripts/run_local.py --max-workloads 3
```

Requires: Local CUDA-capable GPU and `FIB_DATASET_PATH` environment variable.

### Run Benchmarks on Modal (RECOMMENDED)

Test your solution on NVIDIA B200 GPUs via Modal:

**One-time setup:**

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest
```

Note that the path you use in `modal volume put flashinfer-trace /path/to/mlsys26-contest` should ends with `/`.

**Run benchmark:**

```bash
modal run scripts/run_modal.py [Options]
```
Options:
- `--max-workloads`: Run on a subset of workloads for quick testing (default: all workloads)

for example, to run on the first 3 workloads:

```bash
modal run scripts/run_modal.py --max-workloads 3
```

## Profiling
Now the `run_modal.py` script supports profiling and writes outputs to the Modal volume.
```shell
modal run scripts/run_modal.py  \
--max-workloads 1 \
--profile \
--profile-output /data/profiles/fib_profile_modal_trace.json
```
`--profile` enables profiling, and `--profile-output` is the path in Modal volume (`flashinfer-trace`).

After the run, download the profile files from Modal volume to your local machine:

```shell
modal volume get flashinfer-trace --force /profiles/fib_profile_modal_trace.json ./artifacts/fib_profile_modal_trace.json
modal volume get flashinfer-trace --force /profiles/fib_profile_modal_trace.summary.txt ./artifacts/fib_profile_modal_trace.summary.txt
modal volume get flashinfer-trace --force /profiles/fib_profile_modal_trace.phase.json ./artifacts/fib_profile_modal_trace.phase.json
```

Then open Chrome and navigate to `chrome://tracing`, click "Load", and select `fib_profile_modal_trace.json`.

`fib_profile_modal_trace.summary.txt` is a human-readable text summary for quick hotspot inspection.

`fib_profile_modal_trace.phase.json` contains one-shot in-worker phase timing (`phase1`~`phase7`) to help diagnose stage-level bottlenecks when the benchmark runtime uses subprocess execution.

The generated profiling artifacts have different purposes:

- `fib_profile_modal_trace.json`: Full Chrome trace timeline (`chrome://tracing`).
- `fib_profile_modal_trace.summary.txt`: Aggregated hotspot table printed by torch profiler.
- `fib_profile_modal_trace.phase.json`: Stage-level timing split from in-worker CUDA events.

`fib_profile_modal_trace.phase.json` uses the following structure:

```json
{
    "total_tokens": 29120,
    "phase_ms": {
        "phase1_route_permute": 1801.98,
        "phase2_input_permute": 85.31,
        "phase3_gemm1": 716.18,
        "phase4_swiglu": 469.09,
        "phase5_gemm2": 606.58,
        "phase6_scatter": 392.15,
        "phase7_output_cast": 0.34
    },
    "ratio_percent": {
        "phase1_route_permute": 44.26,
        "phase2_input_permute": 2.10,
        "phase3_gemm1": 17.59,
        "phase4_swiglu": 11.52,
        "phase5_gemm2": 14.90,
        "phase6_scatter": 9.63,
        "phase7_output_cast": 0.01
    },
    "total_ms": 4071.63
}
```

Quickly inspect the top phase bottlenecks:

```shell
python - <<'PY'
import json
from pathlib import Path

path = Path("./artifacts/fib_profile_modal_trace.phase.json")
data = json.loads(path.read_text())
ratio = data.get("ratio_percent", {})
phase_ms = data.get("phase_ms", {})

print(f"total_tokens={data.get('total_tokens')}, total_ms={data.get('total_ms'):.3f}")
for name, pct in sorted(ratio.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:24s}  {phase_ms.get(name, 0.0):9.3f} ms  {pct:6.2f}%")
PY
```

## Submission

To submit your solution for evaluation:

1. Ensure your implementation is complete and tested
2. Run `python scripts/pack_solution.py` to generate `solution.json`
3. Commit and push your changes
4. Tag your commit for evaluation (e.g., `git tag submission-v1`)

## Project Structure

```
flashinfer-bench-starter-kit/
├── README.md                    # This file
├── config.toml                  # Track configuration (edit this)
├── solution/                    # Solution source files
│   ├── triton/                  # Triton implementation
│   │   └── kernel.py           # Your Triton kernel
│   └── cuda/                    # CUDA implementation
│       ├── kernel.cu           # Your CUDA kernel
│       └── binding.py          # TVM FFI bindings
├── scripts/                     # Utility scripts
│   ├── run_local.py            # Local benchmark runner
│   ├── run_modal.py            # Modal cloud benchmark runner
│   └── pack_solution.py        # Pack source files into solution.json
└── images/                      # Sponsor logos
```

## Additional Resources

### Solution Handling API

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# Pack source files into a Solution object
spec = BuildSpec(
    language="triton",  # or "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="fused_moe",
    author="your_name",
)

# Extract a Solution to files in a working directory
extract_solution_to_files(solution, "./output_dir")
```

### Running Sanitizers

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

output = flashinfer_bench_run_sanitizer(
    solution=solution,
    workload=workload,
    sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
    timeout=300,
)
print(output)
```

### NCU Profiling

```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu

output = flashinfer_bench_run_ncu(
    solution=solution,
    workload=workload,
    set="detailed",
    page="details",
    timeout=120,
)
print(output)
```

### List Available Tools

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# Returns list of OpenAI-compatible function schemas
```

## Notes

### Kernel Signature Requirements

When implementing kernels using Destination Passing Style (DPS), ensure you specify the kernel signature type in your `BuildSpec` and adjust the build configuration accordingly.

**Important:** Avoid using variadic input arguments in your kernel signatures, as they will fail the builder validation check.

### CUDA Kernel Bindings

For CUDA kernel implementations, we recommend using [TVM FFI](https://tvm.apache.org/ffi/) for Python bindings. The `flashinfer_bench.agents` module provides TVM FFI agent instruction prompts to assist with development.
