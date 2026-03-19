---
name: flashinfer-dev-skills
description: "Standard development process for flashinfer kernel generation contest. Use when the user needs to improve their Triton or CUDA kernel and test."
---

# Development Process
When you work on the kernels, you must follow the "Modify - Pack - Test" steps. Each of them is explained in the following subsections.

## Source Code to Modify
For triton implementation, the source code you need to modify is in `solutions/triton/kernel.py`. The entrypoint is `kernel.py::run`. You MUST NOT change the parameters of the entrypoint, since the benchmark is using this standard entrypoint to evaluate the implementation.

You should always avoid syntax error when coding.

## Pack Solution
After you modify the source code, run `python scripts/pack_solution.py` to pack your solution. You should run this in the correct conda or other environment, like in `fi-bench`, or ask the user.

## Test
Execute
```bash
modal run scripts/run_modal.py [Options]
```
to run test (evaluation) on NVIDIA B200 provided by Modal platform.

Options:
- `--max-workloads`: Run on a subset of workloads for quick testing (default: all workloads)

for example, to run on the first 3 workloads:

```bash
modal run scripts/run_modal.py --max-workloads 3
```

**IMPORTANT**: For test, set `--max-workloads` to a small number (e.g., 3) to run on a subset of workloads for quick testing. This saves time and the user's credit on Modal, which is really expensive. Full evaluation should be executed only by the user.