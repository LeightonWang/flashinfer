import torch
import triton
import triton.language as tl


# ============================================================================
# Persistent Grouped GEMM1 kernel: FP8 A × FP8 B with block-scale dequant
# All experts in a single kernel launch with persistent CTA work scheduling.
#
# Token data is permuted and contiguous per expert via expert_offsets.
# B weights are [E_local, N, K] contiguous (one expert = one slice).
# B scales are [E_local, num_n_blocks, num_k_blocks].
#
# Tile mapping: global tile_id -> (expert_id, m_tile, n_tile) via
# a prefix-sum table (tile_offsets) computed on host.
# ============================================================================
@triton.jit
def _persistent_gemm1_kernel(
    # A (permuted input): [total_tokens, K], fp8
    A_ptr,
    # A scale: [num_k_blocks, total_tokens], fp32
    A_scale_ptr,
    # B weights: [E_local, N, K], fp8 — contiguous per expert
    B_ptr,
    # B scale: [E_local, num_n_blocks, num_k_blocks], fp32
    B_scale_ptr,
    # C output: [total_tokens, N], fp32
    C_ptr,
    # Expert scheduling arrays (device tensors)
    expert_offsets_ptr,   # [E_local + 1], int32 — cumsum of token counts
    tile_offsets_ptr,     # [E_local + 1], int32 — cumsum of tile counts per expert
    num_tiles,            # total number of tiles across all experts
    # Dimensions
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    E_LOCAL: tl.constexpr,
    # Strides
    stride_a_t, stride_a_k,
    stride_as_kb, stride_as_t,
    stride_b_e, stride_b_n, stride_b_k,
    stride_bs_e, stride_bs_nb, stride_bs_kb,
    stride_c_t, stride_c_n,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # Binary search to find which expert this tile belongs to
        # tile_offsets[e] <= tile_id < tile_offsets[e+1]
        lo = 0
        hi = E_LOCAL
        while lo < hi:
            mid = (lo + hi) // 2
            mid_val = tl.load(tile_offsets_ptr + mid + 1)
            if tile_id < mid_val:
                hi = mid
            else:
                lo = mid + 1
        expert_id = lo

        # Tile within this expert
        expert_tile_start = tl.load(tile_offsets_ptr + expert_id)
        tile_in_expert = tile_id - expert_tile_start

        # Expert's token range
        e_tok_start = tl.load(expert_offsets_ptr + expert_id)
        e_tok_end = tl.load(expert_offsets_ptr + expert_id + 1)
        Tk = e_tok_end - e_tok_start

        # Map tile_in_expert -> (m_tile, n_tile)
        num_n_tiles = N // BLOCK_N
        pid_m = tile_in_expert // num_n_tiles
        pid_n = tile_in_expert % num_n_tiles

        # Row/col offsets in the global permuted array
        offs_m = e_tok_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < Tk
        nb = pid_n

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kb in range(NUM_K_BLOCKS):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

            # Load A tile: [BLOCK_M, BLOCK_K], keep fp8 for tensor cores
            a_ptrs = A_ptr + offs_m[:, None] * stride_a_t + offs_k[None, :] * stride_a_k
            a_fp8 = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

            # Load B tile: [BLOCK_N, BLOCK_K], keep fp8 for tensor cores
            b_ptrs = B_ptr + expert_id * stride_b_e + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k
            b_fp8 = tl.load(b_ptrs)

            # FP8 Tensor Core dot + post-hoc scaling
            partial = tl.dot(a_fp8, tl.trans(b_fp8))
            sa = tl.load(A_scale_ptr + kb * stride_as_kb + offs_m * stride_as_t, mask=mask_m, other=0.0)
            sb = tl.load(B_scale_ptr + expert_id * stride_bs_e + nb * stride_bs_nb + kb * stride_bs_kb)
            acc += partial * sa[:, None] * sb

        # Store
        c_ptrs = C_ptr + offs_m[:, None] * stride_c_t + offs_n[None, :] * stride_c_n
        tl.store(c_ptrs, acc, mask=mask_m[:, None])


# ============================================================================
# Persistent Grouped GEMM2 kernel: fp32 A × fp8 B with block-scale dequant
# Same persistent scheduling as GEMM1.
# ============================================================================
@triton.jit
def _persistent_gemm2_kernel(
    A_ptr,                # fp32, [total_tokens, K]
    B_ptr,                # fp8, [E_local, N, K]
    B_scale_ptr,          # fp32, [E_local, num_n_blocks, num_k_blocks]
    C_ptr,                # fp32, [total_tokens, N]
    expert_offsets_ptr,
    tile_offsets_ptr,
    num_tiles,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    E_LOCAL: tl.constexpr,
    stride_a_t, stride_a_k,
    stride_b_e, stride_b_n, stride_b_k,
    stride_bs_e, stride_bs_nb, stride_bs_kb,
    stride_c_t, stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # Binary search for expert
        lo = 0
        hi = E_LOCAL
        while lo < hi:
            mid = (lo + hi) // 2
            mid_val = tl.load(tile_offsets_ptr + mid + 1)
            if tile_id < mid_val:
                hi = mid
            else:
                lo = mid + 1
        expert_id = lo

        expert_tile_start = tl.load(tile_offsets_ptr + expert_id)
        tile_in_expert = tile_id - expert_tile_start

        e_tok_start = tl.load(expert_offsets_ptr + expert_id)
        e_tok_end = tl.load(expert_offsets_ptr + expert_id + 1)
        Tk = e_tok_end - e_tok_start

        num_n_tiles = N // BLOCK_N
        pid_m = tile_in_expert // num_n_tiles
        pid_n = tile_in_expert % num_n_tiles

        offs_m = e_tok_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < Tk
        nb = pid_n

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kb in range(NUM_K_BLOCKS):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

            # A: fp32
            a_ptrs = A_ptr + offs_m[:, None] * stride_a_t + offs_k[None, :] * stride_a_k
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

            # B: fp8 -> fp32
            b_ptrs = B_ptr + expert_id * stride_b_e + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k
            b = tl.load(b_ptrs).to(tl.float32)

            sb = tl.load(B_scale_ptr + expert_id * stride_bs_e + nb * stride_bs_nb + kb * stride_bs_kb)
            b = b * sb

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


def _compute_tile_offsets(expert_counts, BLOCK_M, num_n_tiles, device):
    """Compute prefix-sum of tile counts per expert for persistent scheduling."""
    E_local = expert_counts.numel()
    # Number of M-tiles per expert = ceil(Tk_e / BLOCK_M)
    m_tiles = (expert_counts + BLOCK_M - 1) // BLOCK_M
    tiles_per_expert = m_tiles * num_n_tiles
    tile_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    tile_offsets[1:] = torch.cumsum(tiles_per_expert, dim=0)
    return tile_offsets


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
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

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

    local_expert_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_local)
    token_indices_expanded = torch.arange(T, device=device, dtype=torch.int32).unsqueeze(1).expand(T, TOP_K)

    flat_token_ids = token_indices_expanded[local_expert_mask]
    flat_expert_global = topk_idx[local_expert_mask]
    flat_expert_local = flat_expert_global - local_start

    flat_routing_weights = weights[flat_token_ids.long(), flat_expert_global.long()].to(torch.float32)

    total_tokens = flat_token_ids.numel()

    if total_tokens == 0:
        output.zero_()
        return

    sorted_order = torch.argsort(flat_expert_local.to(torch.int64), stable=True)
    sorted_token_ids = flat_token_ids[sorted_order].to(torch.int32).contiguous()
    sorted_expert_local = flat_expert_local[sorted_order].to(torch.int64).contiguous()
    sorted_weights = flat_routing_weights[sorted_order].contiguous()

    expert_counts = torch.bincount(sorted_expert_local, minlength=E_local).to(torch.int32)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    permuted_input = hidden_states.index_select(0, sorted_token_ids.long()).contiguous()
    permuted_hs_scale = hidden_states_scale.index_select(1, sorted_token_ids.long()).contiguous()

    # ========================================================================
    # Phase 3: Persistent Grouped GEMM1
    # [total_tokens, H] x [E_local, 2*I, H]^T -> [total_tokens, 2*I]
    # Single kernel launch for all experts
    # ========================================================================
    gemm1_out = torch.empty((total_tokens, 2 * I), dtype=torch.float32, device=device)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    num_n_tiles_g1 = (2 * I) // BLOCK_N  # 32
    tile_offsets_g1 = _compute_tile_offsets(expert_counts, BLOCK_M, num_n_tiles_g1, device)
    total_tiles_g1 = int(tile_offsets_g1[-1].item())

    if total_tiles_g1 > 0:
        grid_g1 = (min(NUM_SMS, total_tiles_g1),)
        _persistent_gemm1_kernel[grid_g1](
            permuted_input, permuted_hs_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm1_out,
            expert_offsets, tile_offsets_g1,
            total_tiles_g1,
            K=H, N=2 * I,
            NUM_K_BLOCKS=NUM_H_BLOCKS,
            E_LOCAL=E_local,
            stride_a_t=permuted_input.stride(0), stride_a_k=permuted_input.stride(1),
            stride_as_kb=permuted_hs_scale.stride(0), stride_as_t=permuted_hs_scale.stride(1),
            stride_b_e=gemm1_weights.stride(0), stride_b_n=gemm1_weights.stride(1), stride_b_k=gemm1_weights.stride(2),
            stride_bs_e=gemm1_weights_scale.stride(0), stride_bs_nb=gemm1_weights_scale.stride(1), stride_bs_kb=gemm1_weights_scale.stride(2),
            stride_c_t=gemm1_out.stride(0), stride_c_n=gemm1_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_SMS=NUM_SMS,
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
    # Phase 5: Persistent Grouped GEMM2
    # [total_tokens, I] x [E_local, H, I]^T -> [total_tokens, H]
    # Single kernel launch for all experts
    # ========================================================================
    gemm2_out = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    num_n_tiles_g2 = H // BLOCK_N  # 56
    tile_offsets_g2 = _compute_tile_offsets(expert_counts, BLOCK_M, num_n_tiles_g2, device)
    total_tiles_g2 = int(tile_offsets_g2[-1].item())

    if total_tiles_g2 > 0:
        grid_g2 = (min(NUM_SMS, total_tiles_g2),)
        _persistent_gemm2_kernel[grid_g2](
            swiglu_out,
            gemm2_weights, gemm2_weights_scale,
            gemm2_out,
            expert_offsets, tile_offsets_g2,
            total_tiles_g2,
            K=I, N=H,
            NUM_K_BLOCKS=NUM_I_BLOCKS,
            E_LOCAL=E_local,
            stride_a_t=swiglu_out.stride(0), stride_a_k=swiglu_out.stride(1),
            stride_b_e=gemm2_weights.stride(0), stride_b_n=gemm2_weights.stride(1), stride_b_k=gemm2_weights.stride(2),
            stride_bs_e=gemm2_weights_scale.stride(0), stride_bs_nb=gemm2_weights_scale.stride(1), stride_bs_kb=gemm2_weights_scale.stride(2),
            stride_c_t=gemm2_out.stride(0), stride_c_n=gemm2_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_SMS=NUM_SMS,
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