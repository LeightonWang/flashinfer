import torch
import triton
import triton.language as tl


# ============================================================================
# Per-expert GEMM kernel for FP8 block-scale MoE
# Computes: C[m,n] = sum_k (A_fp8[m,k] * B_fp8[n,k]) * sA[m,kb] * sB[nb,kb]
#
# A = permuted hidden_states for this expert [Tk, K], fp8
# A_scale = [num_k_blocks, total_tokens], fp32 (indexed by global permuted position)
# B = expert weight [N, K], fp8
# B_scale = [num_n_blocks, num_k_blocks], fp32
# C = output for this expert [Tk, N], fp32
# ============================================================================
@triton.jit
def _gemm_fp8_blockscale_kernel(
    A_ptr,              # fp8, base pointer for this expert's input
    A_scale_ptr,        # fp32, [num_k_blocks, total_tokens]
    B_ptr,              # fp8, [N, K] for this expert
    B_scale_ptr,        # fp32, [num_n_blocks, num_k_blocks]
    C_ptr,              # fp32, base pointer for this expert's output
    expert_offset,      # int: starting row in the permuted array for this expert
    Tk,                 # number of tokens for this expert
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    # Strides
    stride_a_t, stride_a_k,
    stride_as_kb, stride_as_t,
    stride_b_n, stride_b_k,
    stride_bs_nb, stride_bs_kb,
    stride_c_t, stride_c_n,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets
    offs_m = expert_offset + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < Tk

    nb = pid_n  # N-block index

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(NUM_K_BLOCKS):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], fp8 -> dequant to fp32
        a_ptrs = A_ptr + offs_m[:, None] * stride_a_t + offs_k[None, :] * stride_a_k
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        # Load and apply A scale: per-token per-K-block
        sa = tl.load(A_scale_ptr + kb * stride_as_kb + offs_m * stride_as_t, mask=mask_m, other=0.0)
        a = a * sa[:, None]

        # Load B tile: [BLOCK_N, BLOCK_K], fp8 -> dequant to fp32
        b_ptrs = B_ptr + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k
        b = tl.load(b_ptrs).to(tl.float32)

        # Load and apply B scale: per-N-block per-K-block (one scalar)
        sb = tl.load(B_scale_ptr + nb * stride_bs_nb + kb * stride_bs_kb)
        b = b * sb

        # GEMM: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

    # Store
    c_ptrs = C_ptr + offs_m[:, None] * stride_c_t + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None])


# ============================================================================
# Per-expert GEMM2 kernel: fp32 A × fp8 B with block-scale dequant
# C[m,n] = sum_k A_fp32[m,k] * (B_fp8[n,k] * sB[nb,kb])
# ============================================================================
@triton.jit 
def _gemm2_fp32a_fp8b_kernel(
    A_ptr,              # fp32, base pointer
    B_ptr,              # fp8, [N, K]
    B_scale_ptr,        # fp32, [num_n_blocks, num_k_blocks]
    C_ptr,              # fp32, base pointer
    expert_offset,
    Tk,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_a_t, stride_a_k,
    stride_b_n, stride_b_k,
    stride_bs_nb, stride_bs_kb,
    stride_c_t, stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = expert_offset + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < Tk
    nb = pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(NUM_K_BLOCKS):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], fp32
        a_ptrs = A_ptr + offs_m[:, None] * stride_a_t + offs_k[None, :] * stride_a_k
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        # Load B tile: [BLOCK_N, BLOCK_K], fp8 -> dequant
        b_ptrs = B_ptr + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k
        b = tl.load(b_ptrs).to(tl.float32)

        # Apply B scale
        sb = tl.load(B_scale_ptr + nb * stride_bs_nb + kb * stride_bs_kb)
        b = b * sb

        # GEMM
        acc += tl.dot(a, tl.trans(b))

    c_ptrs = C_ptr + offs_m[:, None] * stride_c_t + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None])


# ============================================================================
# SwiGLU kernel: out[i] = silu(up[i]) * gate[i]
# Input: [total_tokens, 2*I] (gate = [:, :I], up = [:, I:])
# Output: [total_tokens, I]
# ============================================================================
@triton.jit
def _swiglu_kernel(
    input_ptr,
    output_ptr,
    total_tokens,
    I: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_t = offs_t < total_tokens

    # Load gate (first half) and up (second half)
    gate_ptrs = input_ptr + offs_t[:, None] * (2 * I) + offs_i[None, :]
    up_ptrs = input_ptr + offs_t[:, None] * (2 * I) + (I + offs_i[None, :])

    gate = tl.load(gate_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

    # SwiGLU: silu(up) * gate
    silu_up = up * tl.sigmoid(up)
    result = silu_up * gate

    # Store
    out_ptrs = output_ptr + offs_t[:, None] * I + offs_i[None, :]
    tl.store(out_ptrs, result, mask=mask_t[:, None])


# ============================================================================
# Weighted scatter-add kernel:
# For each permuted token at position p, add weight * value to output[original_token_id]
# ============================================================================
@triton.jit
def _weighted_scatter_add_kernel(
    src_ptr,         # [total_tokens, H], fp32
    dst_ptr,         # [T, H], fp32 (output accumulator)
    token_ids_ptr,   # [total_tokens], int32 - original token indices
    weights_ptr,     # [total_tokens], fp32 - routing weights
    total_tokens,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < total_tokens

    # Load token ids and weights
    tok_ids = tl.load(token_ids_ptr + offs_t, mask=mask_t, other=0)
    w = tl.load(weights_ptr + offs_t, mask=mask_t, other=0.0)

    # Load source values
    src_ptrs = src_ptr + offs_t[:, None] * H + offs_h[None, :]
    vals = tl.load(src_ptrs, mask=mask_t[:, None], other=0.0)

    # Apply weights
    vals = vals * w[:, None]

    # Scatter-add to output (atomic since multiple experts may write same token)
    dst_ptrs = dst_ptr + tok_ids[:, None] * H + offs_h[None, :]
    tl.atomic_add(dst_ptrs, vals, mask=mask_t[:, None])


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
    output: torch.Tensor
):
    # Constants
    H = 7168
    I = 2048
    E_global = 256
    E_local = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    BLOCK = 128
    NUM_H_BLOCKS = H // BLOCK            # 56
    NUM_I_BLOCKS = I // BLOCK            # 16
    NUM_G1_BLOCKS = (2 * I) // BLOCK     # 32

    T = int(routing_logits.shape[0])
    device = hidden_states.device

    # ========================================================================
    # Phase 1: Routing (DeepSeek-V3 no-aux) — PyTorch on CUDA
    # ========================================================================
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E_global // N_GROUP  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    route_mask = torch.zeros_like(s)
    route_mask.scatter_(1, topk_idx, 1.0)
    weights = s * route_mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * float(routed_scaling_factor)

    # ========================================================================
    # Phase 2: Token Permutation — sort tokens by expert
    # ========================================================================
    local_start = int(local_expert_offset)

    # For each token-expert pair, build permutation arrays
    # topk_idx: [T, TOP_K] — which experts each token selected
    # We need to find which of those fall into local expert range [local_start, local_start+E_local)
    local_expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_local)

    # Get (token_id, expert_id) pairs for local experts
    # token_indices_expanded[i,j] = i (token index)
    token_indices_expanded = torch.arange(T, device=device, dtype=torch.int32).unsqueeze(1).expand(T, TOP_K)

    # Filter to local experts only
    local_pairs_mask = local_expert_mask  # [T, TOP_K]
    flat_token_ids = token_indices_expanded[local_pairs_mask]  # [num_pairs]
    flat_expert_global = topk_idx[local_pairs_mask]            # [num_pairs]
    flat_expert_local = flat_expert_global - local_start       # [num_pairs], in [0, E_local)

    # Get routing weights for these pairs
    flat_routing_weights = weights[flat_token_ids.long(), flat_expert_global.long()].to(torch.float32)

    total_tokens = flat_token_ids.numel()

    if total_tokens == 0:
        output.zero_()
        return

    # Sort by local expert id to group tokens by expert
    sorted_order = torch.argsort(flat_expert_local.to(torch.int64), stable=True)
    sorted_token_ids = flat_token_ids[sorted_order].to(torch.int32).contiguous()       # [total_tokens]
    sorted_expert_local = flat_expert_local[sorted_order].to(torch.int64).contiguous() # [total_tokens]
    sorted_weights = flat_routing_weights[sorted_order].contiguous()   # [total_tokens]

    # Compute expert_offsets: cumulative count of tokens per expert
    # expert_offsets[e] = start index, expert_offsets[e+1] = end index
    expert_counts = torch.bincount(sorted_expert_local.to(torch.int64), minlength=E_local).to(torch.int32)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # Permute input: gather rows of hidden_states according to sorted_token_ids
    # permuted_input[i] = hidden_states[sorted_token_ids[i]]
    permuted_input = hidden_states.index_select(0, sorted_token_ids.long()).contiguous()  # [total_tokens, H], fp8
    # Permute hidden_states_scale: scale is [NUM_H_BLOCKS, T], we need [NUM_H_BLOCKS, total_tokens]
    permuted_hs_scale = hidden_states_scale.index_select(1, sorted_token_ids.long()).contiguous()  # [56, total_tokens], fp32

    # ========================================================================
    # Phase 3: Grouped GEMM1 — [total_tokens, H] x [2I, H]^T -> [total_tokens, 2I]
    # Per-expert kernel launches (simpler, avoids complex persistent kernel)
    # ========================================================================
    gemm1_out = torch.empty((total_tokens, 2 * I), dtype=torch.float32, device=device)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    for e in range(E_local):
        tk_e = int(expert_counts[e].item())
        if tk_e == 0:
            continue
        e_offset = int(expert_offsets[e].item())

        grid = (
            (tk_e + BLOCK_M - 1) // BLOCK_M,
            (2 * I) // BLOCK_N,  # 32
        )
        _gemm_fp8_blockscale_kernel[grid](
            permuted_input, permuted_hs_scale,
            gemm1_weights[e], gemm1_weights_scale[e],
            gemm1_out,
            e_offset, tk_e,
            K=H, N=2 * I,
            NUM_K_BLOCKS=NUM_H_BLOCKS,
            stride_a_t=permuted_input.stride(0), stride_a_k=permuted_input.stride(1),
            stride_as_kb=permuted_hs_scale.stride(0), stride_as_t=permuted_hs_scale.stride(1),
            stride_b_n=gemm1_weights[e].stride(0), stride_b_k=gemm1_weights[e].stride(1),
            stride_bs_nb=gemm1_weights_scale[e].stride(0), stride_bs_kb=gemm1_weights_scale[e].stride(1),
            stride_c_t=gemm1_out.stride(0), stride_c_n=gemm1_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

    # ========================================================================
    # Phase 4: SwiGLU — [total_tokens, 4096] -> [total_tokens, 2048]
    # ========================================================================
    swiglu_out = torch.empty((total_tokens, I), dtype=torch.float32, device=device)
    SWIGLU_BLOCK_T = 64
    SWIGLU_BLOCK_I = 128
    grid_swiglu = (
        (total_tokens + SWIGLU_BLOCK_T - 1) // SWIGLU_BLOCK_T,
        I // SWIGLU_BLOCK_I,
    )
    _swiglu_kernel[grid_swiglu](
        gemm1_out, swiglu_out,
        total_tokens, I=I,
        BLOCK_T=SWIGLU_BLOCK_T, BLOCK_I=SWIGLU_BLOCK_I,
        num_warps=4,
    )

    # ========================================================================
    # Phase 5: Grouped GEMM2 — [total_tokens, I] x [H, I]^T -> [total_tokens, H]
    # A = swiglu_out (fp32), B = gemm2_weights (fp8) — no re-quantization
    # ========================================================================
    gemm2_out = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    for e in range(E_local):
        tk_e = int(expert_counts[e].item())
        if tk_e == 0:
            continue
        e_offset = int(expert_offsets[e].item())

        grid = (
            (tk_e + BLOCK_M - 1) // BLOCK_M,
            H // BLOCK_N,  # 56
        )
        _gemm2_fp32a_fp8b_kernel[grid](
            swiglu_out,
            gemm2_weights[e], gemm2_weights_scale[e],
            gemm2_out,
            e_offset, tk_e,
            K=I, N=H,
            NUM_K_BLOCKS=NUM_I_BLOCKS,
            stride_a_t=swiglu_out.stride(0), stride_a_k=swiglu_out.stride(1),
            stride_b_n=gemm2_weights[e].stride(0), stride_b_k=gemm2_weights[e].stride(1),
            stride_bs_nb=gemm2_weights_scale[e].stride(0), stride_bs_kb=gemm2_weights_scale[e].stride(1),
            stride_c_t=gemm2_out.stride(0), stride_c_n=gemm2_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

    # ========================================================================
    # Phase 6: Weighted scatter-add — accumulate into output
    # ========================================================================
    out_accum = torch.zeros((T, H), dtype=torch.float32, device=device)

    SCATTER_BLOCK_T = 64
    SCATTER_BLOCK_H = 128
    grid_scatter = (
        (total_tokens + SCATTER_BLOCK_T - 1) // SCATTER_BLOCK_T,
        H // SCATTER_BLOCK_H,
    )
    _weighted_scatter_add_kernel[grid_scatter](
        gemm2_out, out_accum,
        sorted_token_ids, sorted_weights,
        total_tokens, H=H,
        BLOCK_T=SCATTER_BLOCK_T, BLOCK_H=SCATTER_BLOCK_H,
        num_warps=4,
    )

    output.copy_(out_accum.to(torch.bfloat16))