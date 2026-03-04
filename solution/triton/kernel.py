"""
Fused MoE kernel ported from vLLM's TritonExperts.apply for the
DeepSeek-V3 FP8 block-scale competition.

Entry point: run(routing_logits, routing_bias, hidden_states,
                 hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                 gemm2_weights, gemm2_weights_scale, local_expert_offset,
                 routed_scaling_factor) -> torch.Tensor
"""

from math import prod
from typing import Any

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """Flatten *x* and view as *v* (reuse the underlying storage)."""
    assert prod(v) <= x.numel()
    return x.flatten()[: prod(v)].view(*v)


# ---------------------------------------------------------------------------
# Triton kernels (copied verbatim from vLLM fused_moe.py)
# ---------------------------------------------------------------------------

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,
    stride_bbn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Map program ids to blocks of C (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,
            num_valid_tokens,
        )
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr, stride_cm, stride_cn, pid_n, N,
            offs_token, token_mask,
            BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)

    # Accumulate in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Dequantize
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router weight
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0,
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Python-side kernel launcher (from vLLM invoke_fused_moe_triton_kernel)
# ---------------------------------------------------------------------------

def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
):
    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(
            sorted_token_ids.size(0),
            A.size(0) * top_k * config["BLOCK_SIZE_M"],
        )
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    fused_moe_kernel[grid](
        A,
        B,
        C,
        None,  # B_bias
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        0,  # stride_bbe
        0,  # stride_bbn
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=False,
        HAS_BIAS=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
    )


# ---------------------------------------------------------------------------
# SwiGLU activation: silu(x2) * x1  (matches DeepSeek-V3)
# ---------------------------------------------------------------------------

def swiglu_activation(output: torch.Tensor, input: torch.Tensor) -> None:
    """SwiGLU: split input [..., 2*d] into x1, x2 and compute silu(x2)*x1."""
    torch.ops._C.silu_and_mul(output, input)


# ---------------------------------------------------------------------------
# DeepSeek-V3 no-aux routing
# ---------------------------------------------------------------------------

def deepseek_v3_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-V3 no-aux routing.

    Returns:
        topk_idx:     (T, top_k)   int64 – global expert ids
        topk_weights: (T, top_k)   float32 – normalized routing weights
    """
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)
    E_global = logits.size(1)
    T = logits.size(0)

    # Sigmoid
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    # Group scoring
    group_size = E_global // n_group
    s_wb_grouped = s_with_bias.view(T, n_group, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    # Select top groups
    _, group_idx = torch.topk(
        group_scores, k=topk_group, dim=1, largest=True, sorted=False
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2)
        .expand(T, n_group, group_size)
        .reshape(T, E_global)
    )

    # Global top-k within kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(
        scores_pruned, k=top_k, dim=1, largest=True, sorted=False
    )

    # Combination weights (without bias, normalized, scaled)
    mask = torch.zeros_like(s)
    mask.scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # Gather per-topk weights
    topk_weights = torch.gather(weights, 1, topk_idx)

    return topk_idx, topk_weights


# ---------------------------------------------------------------------------
# Kernel config selection (simplified from vLLM get_default_config)
# ---------------------------------------------------------------------------

def get_kernel_config(
    M: int, E: int, block_shape: list[int],
) -> dict[str, int]:
    """Return Triton kernel tile config for FP8 block-wise quant."""
    return {
        "BLOCK_SIZE_M": 16 if M <= 64 else 64,
        "BLOCK_SIZE_N": block_shape[0],
        "BLOCK_SIZE_K": block_shape[1],
        "GROUP_SIZE_M": 1 if M <= 16 else 32,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 3,
    }


# ---------------------------------------------------------------------------
# Main entry point – competition format
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """
    Fused MoE for DeepSeek-V3 / R1 using vLLM's TritonExperts approach.

    Pipeline:
      1. Routing (DeepSeek-V3 no-aux)
      2. moe_align_block_size (token–expert sorting)
      3. GEMM1 (fused Triton kernel, FP8 block-scale dequant inside kernel)
      4. SwiGLU activation
      5. Dynamic FP8 quantization of intermediate
      6. GEMM2 (fused Triton kernel, with router weight multiplication)
      7. moe_sum (top-k reduce)
    """
    # ----- Fixed DeepSeek-V3/R1 geometry -----
    H = 7168
    I = 2048
    BLOCK = 128
    E_local = gemm1_weights.shape[0]   # 32
    E_global = routing_logits.shape[1]  # 256
    T = routing_logits.shape[0]
    TOP_K = 8
    N = 2 * I  # 4096

    device = hidden_states.device
    block_shape = [BLOCK, BLOCK]

    # =====================================================================
    # Step 1: Routing
    # =====================================================================
    topk_idx, topk_weights = deepseek_v3_routing(
        routing_logits, routing_bias, routed_scaling_factor,
        top_k=TOP_K,
    )
    # topk_idx: (T, 8) global expert ids, topk_weights: (T, 8) float32

    # =====================================================================
    # Step 2: Build expert_map (global -> local)
    # =====================================================================
    expert_map = torch.full(
        (E_global,), -1, dtype=torch.int32, device=device,
    )
    local_start = int(local_expert_offset)
    for i in range(E_local):
        expert_map[local_start + i] = i

    # =====================================================================
    # Step 3: Quantize hidden_states (already FP8) – prepare a1q_scale
    # =====================================================================
    # hidden_states: (T, H) in float8_e4m3fn
    # hidden_states_scale: (H/128, T) – need to transpose to (T, H/128)
    a1q = hidden_states  # already FP8
    a1q_scale = hidden_states_scale.permute(1, 0).contiguous()  # (T, H/128)

    # Weight scales
    w1_scale = gemm1_weights_scale  # (E_local, 2I/128, H/128) = (32, 32, 56)
    w2_scale = gemm2_weights_scale  # (E_local, H/128, I/128) = (32, 56, 16)
    w1 = gemm1_weights              # (E_local, 2I, H) = (32, 4096, 7168) FP8
    w2 = gemm2_weights              # (E_local, H, I) = (32, 7168, 2048) FP8

    # =====================================================================
    # Step 4: Kernel config
    # =====================================================================
    config = get_kernel_config(T, E_local, block_shape)
    compute_type = tl.bfloat16  # FP8 input → BF16 output

    # =====================================================================
    # Step 5: moe_align_block_size (sort tokens by expert)
    # =====================================================================
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_idx.to(torch.int32),
        config["BLOCK_SIZE_M"],
        E_global,
        expert_map,
    )

    # =====================================================================
    # Step 6: Allocate intermediate buffers
    # =====================================================================
    K = H  # hidden dim
    activation_out_dim = N // 2  # SwiGLU: 4096 → 2048

    # workspace2: used for cache1 (GEMM1 output) then reused for cache3
    workspace2 = torch.empty(
        T * TOP_K * max(N, K),
        device=device, dtype=torch.bfloat16,
    )
    # workspace13: used for cache2 (activation output)
    workspace13 = torch.empty(
        T * TOP_K * max(activation_out_dim, K),
        device=device, dtype=torch.bfloat16,
    )

    intermediate_cache1 = _resize_cache(workspace2, (T, TOP_K, N))
    intermediate_cache2 = _resize_cache(
        workspace13, (T * TOP_K, activation_out_dim),
    )
    intermediate_cache3 = _resize_cache(workspace2, (T, TOP_K, K))

    output = torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # =====================================================================
    # Step 7: GEMM1 – all experts fused, FP8 block-scale dequant in kernel
    # =====================================================================
    invoke_fused_moe_triton_kernel(
        A=a1q,                          # (T, H) FP8
        B=w1,                           # (E_local, 2I, H) FP8
        C=intermediate_cache1,          # (T, TOP_K, 2I) BF16
        A_scale=a1q_scale,              # (T, H/128)
        B_scale=w1_scale,               # (E_local, 2I/128, H/128)
        topk_weights=None,              # don't multiply router weight here
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=TOP_K,
        config=config,
        compute_type=compute_type,
        use_fp8_w8a8=True,
        per_channel_quant=False,
        block_shape=block_shape,
    )

    # =====================================================================
    # Step 8: SwiGLU activation
    # =====================================================================
    swiglu_activation(intermediate_cache2, intermediate_cache1.view(-1, N))

    # =====================================================================
    # Step 9: Dynamic FP8 quantization of intermediate activation
    # =====================================================================
    qintermediate_cache2, a2q_scale = per_token_group_quant_fp8(
        intermediate_cache2, BLOCK,
    )

    # =====================================================================
    # Step 10: GEMM2 – with router weight multiplication fused in kernel
    # =====================================================================
    invoke_fused_moe_triton_kernel(
        A=qintermediate_cache2,         # (T*TOP_K, I) FP8
        B=w2,                           # (E_local, H, I) FP8
        C=intermediate_cache3,          # (T, TOP_K, H) BF16
        A_scale=a2q_scale,              # (T*TOP_K, I/128)
        B_scale=w2_scale,               # (E_local, H/128, I/128)
        topk_weights=topk_weights,      # (T, 8) – multiply router weight
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,                        # top_k=1 for second GEMM (see vLLM)
        config=config,
        compute_type=compute_type,
        use_fp8_w8a8=True,
        per_channel_quant=False,
        block_shape=block_shape,
    )

    # =====================================================================
    # Step 11: Reduce across top-k experts → final output
    # =====================================================================
    ops.moe_sum(
        intermediate_cache3.view(T, TOP_K, K),
        output,
    )

    return output
