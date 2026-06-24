"""
Standalone correctness test + benchmark for solution/triton/kernel.py.

This script does NOT depend on flashinfer / flashinfer-bench. It only needs
Python + torch + triton + a CUDA GPU. It can run in two input modes:

1. Dataset mode (default when a dataset is found): reads the real workloads from
   the mlsys26-contest dataset (workloads/moe/*.jsonl + definitions/moe/*.json),
   materializing inputs exactly the way flashinfer-bench does:
     - "safetensors" inputs  -> loaded from the .safetensors blob files
       (parsed with a tiny built-in reader; no `safetensors` package needed).
     - "scalar" inputs       -> taken directly from the workload spec.
     - "random" inputs       -> generated with torch.randn (fp8 clamped to
       [-2, 2]), mirroring flashinfer_bench.bench.utils._rand_tensor.
   If a safetensors blob is missing (e.g. Git-LFS pointer not pulled) the
   corresponding tensor falls back to random generation with a warning, so the
   script still runs end-to-end.

2. Synthetic mode (--synthetic): builds fully synthetic inputs for arbitrary
   token counts via --sizes.

In both modes it computes a pure-torch reference, verifies the Triton kernel's
output, and benchmarks its latency.

Usage:
    python scripts/standalone_test.py                       # dataset mode, all workloads
    python scripts/standalone_test.py --dataset /path/to/mlsys26-contest
    python scripts/standalone_test.py --workload-index 0 2  # only some workloads
    python scripts/standalone_test.py --synthetic --sizes 256 1024 4096
    python scripts/standalone_test.py --no-benchmark        # correctness only
    python scripts/standalone_test.py --no-test             # benchmark only
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch

# Make `import kernel` resolve to solution/triton/kernel.py
PROJECT_ROOT = Path(__file__).parent.parent
TRITON_DIR = PROJECT_ROOT / "solution" / "triton"
sys.path.insert(0, str(TRITON_DIR))

# `kernel` (and thus triton) is imported lazily inside run_kernel() so that the
# data-loading paths can be exercised without a GPU / triton install.
_K = None


def _get_kernel():
    global _K
    if _K is None:
        import kernel as K  # noqa: E402

        _K = K
    return _K

# ---------------------------------------------------------------------------
# Track constants (must match solution/triton/kernel.py::run)
# ---------------------------------------------------------------------------
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128
GROUP_SIZE = E_GLOBAL // N_GROUP        # 32
NUM_H_BLOCKS = H // BLOCK               # 56
NUM_2I_BLOCKS = (2 * I) // BLOCK        # 32
NUM_I_BLOCKS = I // BLOCK               # 16

FP8_DTYPE = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Synthetic input construction
# ---------------------------------------------------------------------------
def make_inputs(T: int, device: torch.device, seed: int = 0):
    """Build synthetic inputs matching the kernel's expected layout/dtypes."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    def rand(*shape):
        return torch.rand(*shape, generator=g, device=device, dtype=torch.float32)

    routing_logits = randn(T, E_GLOBAL)
    routing_bias = randn(E_GLOBAL)

    # FP8 activations / weights: keep magnitudes small to stay in e4m3 range.
    hidden_states = (randn(T, H) * 0.25).to(FP8_DTYPE)
    # hidden_states_scale layout: [num_k_blocks, T]
    hidden_states_scale = (rand(NUM_H_BLOCKS, T) * 0.02 + 0.01)

    gemm1_weights = (randn(E_LOCAL, 2 * I, H) * 0.25).to(FP8_DTYPE)
    gemm1_weights_scale = (rand(E_LOCAL, NUM_2I_BLOCKS, NUM_H_BLOCKS) * 0.02 + 0.01)

    gemm2_weights = (randn(E_LOCAL, H, I) * 0.25).to(FP8_DTYPE)
    gemm2_weights_scale = (rand(E_LOCAL, NUM_H_BLOCKS, NUM_I_BLOCKS) * 0.02 + 0.01)

    local_expert_offset = 0
    routed_scaling_factor = 2.5

    output = torch.empty(T, H, dtype=torch.bfloat16, device=device)

    return dict(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
        output=output,
    )


# ---------------------------------------------------------------------------
# Dataset (mlsys26-contest) input construction
# ---------------------------------------------------------------------------
DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"


def _input_dtype(name):
    return {
        "routing_logits": torch.float32,
        "routing_bias": torch.bfloat16,
        "hidden_states": FP8_DTYPE,
        "hidden_states_scale": torch.float32,
        "gemm1_weights": FP8_DTYPE,
        "gemm1_weights_scale": torch.float32,
        "gemm2_weights": FP8_DTYPE,
        "gemm2_weights_scale": torch.float32,
    }[name]


def _input_shape(name, T):
    return {
        "routing_logits": (T, E_GLOBAL),
        "routing_bias": (E_GLOBAL,),
        "hidden_states": (T, H),
        "hidden_states_scale": (NUM_H_BLOCKS, T),
        "gemm1_weights": (E_LOCAL, 2 * I, H),
        "gemm1_weights_scale": (E_LOCAL, NUM_2I_BLOCKS, NUM_H_BLOCKS),
        "gemm2_weights": (E_LOCAL, H, I),
        "gemm2_weights_scale": (E_LOCAL, NUM_H_BLOCKS, NUM_I_BLOCKS),
    }[name]


# safetensors dtype string -> (torch dtype, struct/byte info) for the built-in reader.
_ST_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _load_safetensors(path: Path, keys=None):
    """Minimal safetensors reader: returns {key: cpu tensor}.

    Avoids depending on the `safetensors` package. Raises FileNotFoundError if
    the file is missing and ValueError if it is an un-pulled Git-LFS pointer.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len > (1 << 30):
            # Not a real safetensors file; likely an LFS pointer text file.
            f.seek(0)
            if f.read(len(_LFS_MAGIC)) == _LFS_MAGIC:
                raise ValueError(f"{path} is a Git-LFS pointer (run `git lfs pull`).")
            raise ValueError(f"{path} is not a valid safetensors file.")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        out = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if keys is not None and key not in keys:
                continue
            dtype = _ST_DTYPES[meta["dtype"]]
            shape = meta["shape"]
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            # Interpret raw bytes as uint8, then bit-cast to the target dtype.
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype)
            out[key] = t.reshape(shape).clone()
        return out


def _rand_tensor(shape, dtype, device, generator):
    """Mirror flashinfer_bench.bench.utils._rand_tensor (random workload inputs)."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = torch.randn(shape, dtype=torch.float32, device=device,
                        generator=generator).clamp_(-2.0, 2.0)
        return t.to(dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device,
                             generator=generator)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype,
                             generator=generator)
    raise ValueError(f"Unsupported random dtype: {dtype}")


def find_dataset(explicit=None):
    """Locate the mlsys26-contest dataset root."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("FIB_DATASET_PATH"):
        candidates.append(Path(os.environ["FIB_DATASET_PATH"]))
    candidates += [
        Path.home() / "dev" / "mlsys26-contest",
        PROJECT_ROOT.parent / "mlsys26-contest",
    ]
    for c in candidates:
        if c and (c / "workloads" / "moe" / f"{DEFINITION_NAME}.jsonl").exists():
            return c
    return None


def load_workloads(dataset_root: Path):
    """Parse the moe workload jsonl into a list of dicts."""
    jsonl = dataset_root / "workloads" / "moe" / f"{DEFINITION_NAME}.jsonl"
    entries = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def make_inputs_from_workload(entry, dataset_root, device, seed=0):
    """Materialize one workload's inputs exactly like flashinfer-bench.

    safetensors inputs are loaded from disk (falling back to random if the blob
    is missing); scalar inputs are taken from the spec; everything else is
    random (torch.randn-based).
    """
    wl = entry["workload"]
    T = int(wl["axes"]["seq_len"])
    specs = wl["inputs"]
    g = torch.Generator(device=device).manual_seed(seed)

    # Cache safetensors files so we read each blob once.
    st_cache = {}

    def get_input(name):
        spec = specs[name]
        kind = spec["type"]
        dtype = _input_dtype(name)
        shape = _input_shape(name, T)
        if kind == "safetensors":
            path = spec["path"]
            p = Path(path)
            if not p.is_absolute():
                p = (dataset_root / path).resolve()
            try:
                if str(p) not in st_cache:
                    st_cache[str(p)] = _load_safetensors(p)
                t = st_cache[str(p)][spec["tensor_key"]]
                if tuple(t.shape) != tuple(shape):
                    raise ValueError(
                        f"'{name}' expected {tuple(shape)}, got {tuple(t.shape)}")
                return t.to(device=device, dtype=dtype)
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"[warn ] safetensors for '{name}' unavailable "
                      f"({e}); falling back to random.")
                return _rand_tensor(shape, dtype, device, g)
        if kind == "scalar":
            return spec["value"]
        # random
        return _rand_tensor(shape, dtype, device, g)

    args = {name: get_input(name) for name in (
        "routing_logits", "routing_bias", "hidden_states", "hidden_states_scale",
        "gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale",
    )}
    args["local_expert_offset"] = int(specs["local_expert_offset"]["value"])
    args["routed_scaling_factor"] = float(specs["routed_scaling_factor"]["value"])
    args["output"] = torch.empty(T, H, dtype=torch.bfloat16, device=device)
    args["_T"] = T
    args["_uuid"] = wl.get("uuid", "")
    return args


# ---------------------------------------------------------------------------
# Pure-torch reference (no flashinfer / no triton)
# ---------------------------------------------------------------------------
def _route_reference(routing_logits, routing_bias, routed_scaling_factor,
                     local_expert_offset):
    """DeepSeek-style grouped top-k routing, returns per-(token,k) local
    assignment and weight, mirroring _route_select_local_kernel."""
    T = routing_logits.shape[0]
    s = torch.sigmoid(routing_logits.float())            # [T, E]
    sb = s + routing_bias.float()                        # [T, E]

    # Per-group score = sum of top-2 (s+bias) within the group.
    sb_g = sb.view(T, N_GROUP, GROUP_SIZE)
    top2 = torch.topk(sb_g, 2, dim=2).values             # [T, N_GROUP, 2]
    group_scores = top2.sum(dim=2)                       # [T, N_GROUP]

    # Select TOPK_GROUP groups; mask experts in unselected groups.
    sel_groups = torch.topk(group_scores, TOPK_GROUP, dim=1).indices  # [T, TOPK_GROUP]
    group_mask = torch.zeros(T, N_GROUP, dtype=torch.bool, device=s.device)
    group_mask.scatter_(1, sel_groups, True)
    expert_in_sel = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_GLOBAL)

    pruned = torch.where(expert_in_sel, sb, torch.full_like(sb, float("-inf")))

    # Select TOP_K experts by (s+bias) among pruned candidates.
    topk_ids = torch.topk(pruned, TOP_K, dim=1).indices  # [T, TOP_K]
    topk_s = torch.gather(s, 1, topk_ids)                # original sigmoid (no bias)
    denom = topk_s.sum(dim=1, keepdim=True)
    weights = topk_s * routed_scaling_factor / (denom + 1e-20)  # [T, TOP_K]

    is_local = (topk_ids >= local_expert_offset) & (topk_ids < local_expert_offset + E_LOCAL)
    local_ids = topk_ids - local_expert_offset
    return local_ids, weights, is_local


def reference(args) -> torch.Tensor:
    """Full pure-torch reference computation. Returns bf16 [T, H]."""
    device = args["hidden_states"].device
    T = args["routing_logits"].shape[0]

    local_ids, weights, is_local = _route_reference(
        args["routing_logits"], args["routing_bias"],
        args["routed_scaling_factor"], args["local_expert_offset"],
    )

    out = torch.zeros(T, H, dtype=torch.float32, device=device)

    hs = args["hidden_states"].float()                       # [T, H]
    hs_scale = args["hidden_states_scale"].float()           # [NUM_H_BLOCKS, T]
    # Dequant activations: a_deq[i,k] = a_fp8[i,k] * scale[k//128, i]
    # (block scale is constant within a block, so elementwise dequant is exact)
    hs_scale_full = hs_scale.repeat_interleave(BLOCK, dim=0).t()  # [T, H]
    hs_deq = hs * hs_scale_full                              # [T, H]

    g1_scale = args["gemm1_weights_scale"].float()           # [E, 32, 56]
    g2_scale = args["gemm2_weights_scale"].float()           # [E, 56, 16]

    # token -> list of (expert, weight)
    flat_local = is_local
    for e in range(E_LOCAL):
        # tokens routed to this local expert (any of its TOP_K slots)
        sel = flat_local & (local_ids == e)                  # [T, TOP_K]
        tok_idx, k_idx = torch.nonzero(sel, as_tuple=True)
        if tok_idx.numel() == 0:
            continue
        w = weights[tok_idx, k_idx]                          # [n_e]

        a = hs_deq[tok_idx]                                  # [n_e, H]

        # GEMM1: dequant weights, [2I, H]
        b1 = args["gemm1_weights"][e].float()                # [2I, H]
        b1_scale_full = g1_scale[e].repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)  # [2I, H]
        b1_deq = b1 * b1_scale_full
        g1 = a @ b1_deq.t()                                  # [n_e, 2I]

        # SwiGLU: gate = [:, :I], up = [:, I:]; silu(up) * gate
        gate = g1[:, :I]
        up = g1[:, I:]
        act = (up * torch.sigmoid(up)) * gate                # [n_e, I]

        # GEMM2: A=fp32 act (no scale), B fp8 dequant [H, I]
        b2 = args["gemm2_weights"][e].float()                # [H, I]
        b2_scale_full = g2_scale[e].repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)  # [H, I]
        b2_deq = b2 * b2_scale_full
        g2 = act @ b2_deq.t()                                # [n_e, H]

        # Weighted scatter-add
        out.index_add_(0, tok_idx, g2 * w[:, None])

    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Kernel invocation
# ---------------------------------------------------------------------------
def run_kernel(args):
    K = _get_kernel()
    K.run(
        routing_logits=args["routing_logits"],
        routing_bias=args["routing_bias"],
        hidden_states=args["hidden_states"],
        hidden_states_scale=args["hidden_states_scale"],
        gemm1_weights=args["gemm1_weights"],
        gemm1_weights_scale=args["gemm1_weights_scale"],
        gemm2_weights=args["gemm2_weights"],
        gemm2_weights_scale=args["gemm2_weights_scale"],
        local_expert_offset=args["local_expert_offset"],
        routed_scaling_factor=args["routed_scaling_factor"],
        output=args["output"],
    )


# ---------------------------------------------------------------------------
# Correctness + benchmark
# ---------------------------------------------------------------------------
def check_correctness(args, label, atol, rtol, matched_ratio_thresh):
    run_kernel(args)
    torch.cuda.synchronize(args["output"].device)

    got = args["output"].float()
    ref = reference(args).float()

    # Mirror flashinfer-bench LowBitEvaluator / compute_error_stats:
    # an element fails only if it exceeds BOTH atol and rtol; PASS requires
    # matched_ratio >= required_matched_ratio (0.95 for moe_fp8_block_scale).
    abs_err = (got - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-8)
    exceeds = (abs_err > atol) & (rel_err > rtol)
    matched_ratio = 1.0 - exceeds.float().mean().item()
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()

    ok = matched_ratio >= matched_ratio_thresh
    status = "PASS" if ok else "FAIL"
    print(f"[test ] {label} | {status} | matched={matched_ratio*100:.2f}% "
          f"(thresh {matched_ratio_thresh*100:.0f}%) | max_abs_err={max_abs:.3e} | "
          f"max_rel_err={max_rel:.3e}")
    return ok


def _time_fn(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)

    return start.elapsed_time(end) / iters


def benchmark(args, label, device, warmup, iters):
    kernel_ms = _time_fn(lambda: run_kernel(args), device, warmup, iters)
    ref_ms = _time_fn(lambda: reference(args), device, warmup, iters)
    speedup = ref_ms / kernel_ms if kernel_ms > 0 else float("nan")

    print(f"[bench] {label} | kernel={kernel_ms:.3f} ms/iter | "
          f"reference={ref_ms:.3f} ms/iter | speedup={speedup:.2f}x | "
          f"({iters} iters, {warmup} warmup)")
    return kernel_ms, ref_ms, speedup


def profile_run(args, label, device, warmup, iters, trace_path):
    """Profile the Triton kernel with torch.profiler and export a chrome trace.

    The kernel sets per-phase nvtx/record_function ranges when
    FIB_PROFILE_PHASES=1 (set in main before importing/launching), so the trace
    will contain phase1_route_permute ... phase7_output_cast spans.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        run_kernel(args)
    torch.cuda.synchronize(device)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False,
                 with_stack=False) as prof:
        for _ in range(iters):
            run_kernel(args)
        torch.cuda.synchronize(device)

    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    print(f"\n[profile] {label} | trace -> {trace_path}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=15))


def main():
    parser = argparse.ArgumentParser(
        description="Standalone (torch+triton only) test & benchmark for the Triton MoE kernel."
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the mlsys26-contest dataset root. If omitted, "
                             "auto-detected via $FIB_DATASET_PATH and common locations.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use fully synthetic inputs (--sizes) instead of the dataset.")
    parser.add_argument("--workload-index", type=int, nargs="+", default=None,
                        help="Dataset mode: only run these workload indices (0-based).")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024],
                        help="Synthetic mode: token counts (T) to test/benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.3)
    parser.add_argument("--matched-ratio", type=float, default=0.9,
                        help="Required fraction of elements within tolerance. "
                             "The moe_fp8_block_scale evaluation config is "
                             "atol=1.0, rtol=0.3, required_matched_ratio=0.9.")
    parser.add_argument("--no-test", action="store_true", help="Skip correctness check.")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmark.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile the kernel and export a chrome trace.")
    parser.add_argument("--profile-iters", type=int, default=5,
                        help="Iterations captured in the profile trace.")
    parser.add_argument("--trace-output", type=str, default="artifacts/standalone_trace.json",
                        help="Chrome trace output path (load via chrome://tracing).")
    args = parser.parse_args()

    if args.profile:
        # Enable per-phase nvtx/record_function ranges inside kernel.py.
        os.environ["FIB_PROFILE_PHASES"] = "1"

    if not torch.cuda.is_available():
        print("CUDA device required.", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)} | "
          f"torch {torch.__version__}")

    # Decide input mode: dataset (default) vs synthetic.
    dataset_root = None if args.synthetic else find_dataset(args.dataset)

    if dataset_root is not None:
        print(f"Dataset: {dataset_root}")
        entries = load_workloads(dataset_root)
        indices = args.workload_index if args.workload_index is not None \
            else list(range(len(entries)))
        cases = []
        for idx in indices:
            entry = entries[idx]
            built = make_inputs_from_workload(entry, dataset_root, device, args.seed)
            label = (f"#{idx:<2} T={built['_T']:>6} "
                     f"off={built['local_expert_offset']:>3}")
            cases.append((label, built))
    else:
        if not args.synthetic and args.dataset is None:
            print("[warn ] dataset not found; using synthetic inputs. "
                  "Pass --dataset PATH or set $FIB_DATASET_PATH for real data.")
        cases = []
        for T in args.sizes:
            built = make_inputs(T, device, args.seed)
            cases.append((f"T={T:>6}", built))

    all_pass = True
    for label, built in cases:
        if not args.no_test:
            ok = check_correctness(built, label, args.atol, args.rtol,
                                   args.matched_ratio)
            all_pass = all_pass and ok
        if not args.no_benchmark:
            benchmark(built, label, device, args.warmup, args.iters)
        if args.profile:
            stem = Path(args.trace_output)
            out = stem if len(cases) == 1 else stem.with_name(
                f"{stem.stem}_{label.split()[0].strip('#')}{stem.suffix}")
            profile_run(built, label, device, args.warmup, args.profile_iters, out)

    if not args.no_test and not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
