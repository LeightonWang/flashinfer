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
    STORE_BF16: tl.constexpr,
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
        mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < Tk
        nb = pid_n

        # Create block pointers for TMA (Tensor Memory Accelerator) loads
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr + e_tok_start * stride_a_t,
            shape=(Tk, K),
            strides=(stride_a_t, stride_a_k),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr + expert_id * stride_b_e,
            shape=(N, K),
            strides=(stride_b_n, stride_b_k),
            offsets=(pid_n * BLOCK_N, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0)
        )

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kb in range(NUM_K_BLOCKS):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

            # Load A tile: TMA block load (asynchronous hardware copy)
            a_fp8 = tl.load(a_block_ptr, boundary_check=(0, 1))

            # Load B tile: TMA block load
            b_fp8 = tl.load(b_block_ptr, boundary_check=(0, 1))

            # FP8 Tensor Core dot + post-hoc scaling
            partial = tl.dot(a_fp8, tl.trans(b_fp8))
            sa = tl.load(A_scale_ptr + kb * stride_as_kb + offs_m * stride_as_t, mask=mask_m, other=0.0)
            sb = tl.load(B_scale_ptr + expert_id * stride_bs_e + nb * stride_bs_nb + kb * stride_bs_kb)
            acc += partial * sa[:, None] * sb
            
            # Advance TMA pointers to the next K block
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

        # Store C tile using TMA
        c_block_ptr = tl.make_block_ptr(
            base=C_ptr + e_tok_start * stride_c_t,
            shape=(Tk, N),
            strides=(stride_c_t, stride_c_n),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        if STORE_BF16:
            tl.store(c_block_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))
        else:
            tl.store(c_block_ptr, acc, boundary_check=(0, 1))


# ============================================================================
# Persistent Grouped GEMM2 kernel: bf16/fp32 A × fp8 B with block-scale dequant
# Same persistent scheduling as GEMM1.
# ============================================================================
@triton.jit
def _persistent_gemm2_kernel(
    A_ptr,                # bf16/fp32, [total_tokens, K]
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
    GEMM2_USE_BF16_DOT: tl.constexpr,
    STORE_BF16: tl.constexpr,
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
        mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < Tk
        nb = pid_n

        # Create block pointers for TMA (Tensor Memory Accelerator) loads
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr + e_tok_start * stride_a_t,
            shape=(Tk, K),
            strides=(stride_a_t, stride_a_k),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr + expert_id * stride_b_e,
            shape=(N, K),
            strides=(stride_b_n, stride_b_k),
            offsets=(pid_n * BLOCK_N, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0)
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kb in range(NUM_K_BLOCKS):
            # A: bf16/fp32, TMA load
            a = tl.load(a_block_ptr, boundary_check=(0, 1))

            # B: fp8, TMA load
            b_fp8 = tl.load(b_block_ptr, boundary_check=(0, 1))

            sb = tl.load(B_scale_ptr + expert_id * stride_bs_e + nb * stride_bs_nb + kb * stride_bs_kb)

            # Prefer BF16 Tensor Core dot on Blackwell; keep TF32 fallback for accuracy checks.
            if GEMM2_USE_BF16_DOT:
                partial = tl.dot(a.to(tl.bfloat16), tl.trans(b_fp8.to(tl.bfloat16)), out_dtype=tl.float32)
            else:
                partial = tl.dot(a.to(tl.float32), tl.trans(b_fp8.to(tl.float32)), input_precision="tf32")

            # sb is a scalar for this (expert, n-block, k-block), so scaling after dot
            # avoids per-element matrix scaling overhead in the hot loop.
            acc += partial * sb

            # Advance TMA pointers to the next K block
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

        # Store C tile using TMA
        c_block_ptr = tl.make_block_ptr(
            base=C_ptr + e_tok_start * stride_c_t,
            shape=(Tk, N),
            strides=(stride_c_t, stride_c_n),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        if STORE_BF16:
            tl.store(c_block_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))
        else:
            tl.store(c_block_ptr, acc, boundary_check=(0, 1))


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
    STORE_BF16: tl.constexpr,
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
    if STORE_BF16:
        tl.store(out_ptrs, result.to(tl.bfloat16), mask=mask_t[:, None])
    else:
        tl.store(out_ptrs, result, mask=mask_t[:, None])


# ============================================================================
# Weighted scatter-add kernel:
# For each permuted token at position p, add weight * value to output[original_token_id]
# ============================================================================
@triton.jit
def _weighted_scatter_add_kernel(
    src_ptr,         # [total_tokens, H], bf16/fp32
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
    vals = tl.load(src_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

    # Apply weights
    vals = vals * w[:, None]

    # Scatter-add to output (atomic since multiple experts may write same token)
    dst_ptrs = dst_ptr + tok_ids[:, None] * H + offs_h[None, :]
    tl.atomic_add(dst_ptrs, vals, mask=mask_t[:, None])


@triton.jit
def _route_and_bucket_local_kernel(
    routing_logits_ptr,      # [T, E_global], fp32/fp16/bf16
    routing_bias_ptr,        # [E_global], fp32
    expert_counts_ptr,       # [E_local], int32
    bucket_token_ids_ptr,    # [E_local, T], int32
    bucket_weights_ptr,      # [E_local, T], fp32
    T,
    stride_rl_t, stride_rl_e,
    stride_bucket_tok_e, stride_bucket_tok_t,
    stride_bucket_w_e, stride_bucket_w_t,
    local_start,
    routed_scaling_factor,
    E_GLOBAL: tl.constexpr,
    E_LOCAL: tl.constexpr,
    TOP_K: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    neg_inf_e = tl.full((E_GLOBAL,), -float("inf"), dtype=tl.float32)
    neg_inf_g = tl.full((N_GROUP,), -float("inf"), dtype=tl.float32)

    e = tl.arange(0, E_GLOBAL)
    groups = tl.arange(0, N_GROUP)
    k_arange = tl.arange(0, TOP_K)

    logits = tl.load(routing_logits_ptr + pid * stride_rl_t + e * stride_rl_e).to(tl.float32)
    bias = tl.load(routing_bias_ptr + e).to(tl.float32)

    # DeepSeek-style grouped routing: compute top2 per group via 2D reductions.
    s_with_bias = tl.sigmoid(logits) + bias
    s_with_bias_2d = tl.reshape(s_with_bias, (N_GROUP, GROUP_SIZE))

    group_max1 = tl.max(s_with_bias_2d, axis=1)
    group_idx1 = tl.argmax(s_with_bias_2d, axis=1)

    cols = tl.arange(0, GROUP_SIZE)
    s_with_bias_wo_top1 = tl.where(
        cols[None, :] != group_idx1[:, None], s_with_bias_2d, -float("inf")
    )
    group_max2 = tl.max(s_with_bias_wo_top1, axis=1)
    group_scores = group_max1 + group_max2

    # Keep only experts in selected groups.
    selected_groups = tl.zeros((N_GROUP,), dtype=tl.int32)
    for _ in range(TOPK_GROUP):
        masked_group_scores = tl.where(selected_groups > 0, neg_inf_g, group_scores)
        g_sel = tl.argmax(masked_group_scores, axis=0)
        selected_groups = selected_groups + (groups == g_sel).to(tl.int32)

    pruned_scores_2d = tl.where(selected_groups[:, None] > 0, s_with_bias_2d, -float("inf"))
    pruned_scores = tl.reshape(pruned_scores_2d, (E_GLOBAL,))

    selected_experts = tl.zeros((E_GLOBAL,), dtype=tl.int32)
    topk_ids = tl.zeros((TOP_K,), dtype=tl.int32)
    topk_s = tl.zeros((TOP_K,), dtype=tl.float32)
    for k in range(TOP_K):
        masked_scores = tl.where(selected_experts > 0, neg_inf_e, pruned_scores)
        e_sel = tl.argmax(masked_scores, axis=0)
        selected_experts = selected_experts + (e == e_sel).to(tl.int32)

        # Scalar load avoids scanning E_GLOBAL to fetch sigmoid score of selected expert.
        logit_sel = tl.load(
            routing_logits_ptr + pid * stride_rl_t + e_sel * stride_rl_e
        ).to(tl.float32)
        s_sel = tl.sigmoid(logit_sel)
        topk_ids = tl.where(k_arange == k, e_sel.to(tl.int32), topk_ids)
        topk_s = tl.where(k_arange == k, s_sel, topk_s)

    denom = tl.sum(topk_s, axis=0)

    inv_denom = routed_scaling_factor / (denom + 1e-20)

    for k in range(TOP_K):
        k_mask = k_arange == k
        expert_id = tl.sum(tl.where(k_mask, topk_ids, 0), axis=0)
        topk_s_k = tl.sum(tl.where(k_mask, topk_s, 0.0), axis=0)
        is_local = (expert_id >= local_start) & (expert_id < local_start + E_LOCAL)
        local_id = (expert_id - local_start).to(tl.int32)
        local_id_safe = tl.where(is_local, local_id, 0)
        w = (topk_s_k * inv_denom).to(tl.float32)

        pos = tl.atomic_add(expert_counts_ptr + local_id_safe, 1, mask=is_local)

        tok_ptr = (
            bucket_token_ids_ptr
            + local_id_safe * stride_bucket_tok_e
            + pos * stride_bucket_tok_t
        )
        w_ptr = (
            bucket_weights_ptr
            + local_id_safe * stride_bucket_w_e
            + pos * stride_bucket_w_t
        )

        tl.store(tok_ptr, pid.to(tl.int32), mask=is_local)
        tl.store(w_ptr, w, mask=is_local)


@triton.jit
def _compact_expert_buckets_kernel(
    bucket_token_ids_ptr,  # [E_local, T], int32
    bucket_weights_ptr,    # [E_local, T], fp32
    expert_counts_ptr,     # [E_local], int32
    expert_offsets_ptr,    # [E_local + 1], int32
    sorted_token_ids_ptr,  # [total_tokens], int32
    sorted_weights_ptr,    # [total_tokens], fp32
    stride_bucket_tok_e, stride_bucket_tok_t,
    stride_bucket_w_e, stride_bucket_w_t,
    E_LOCAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)

    if pid_e >= E_LOCAL:
        return

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    count = tl.load(expert_counts_ptr + pid_e)
    mask = offs_t < count

    src_tok_ptrs = (
        bucket_token_ids_ptr
        + pid_e * stride_bucket_tok_e
        + offs_t * stride_bucket_tok_t
    )
    src_w_ptrs = (
        bucket_weights_ptr
        + pid_e * stride_bucket_w_e
        + offs_t * stride_bucket_w_t
    )

    tok = tl.load(src_tok_ptrs, mask=mask, other=0)
    w = tl.load(src_w_ptrs, mask=mask, other=0.0)

    dst_base = tl.load(expert_offsets_ptr + pid_e)
    dst_offs = dst_base + offs_t

    tl.store(sorted_token_ids_ptr + dst_offs, tok, mask=mask)
    tl.store(sorted_weights_ptr + dst_offs, w, mask=mask)


def _compute_tile_offsets(expert_counts, BLOCK_M, num_n_tiles, device):
    """Compute prefix-sum of tile counts per expert for persistent scheduling."""
    E_local = expert_counts.numel()
    # Number of M-tiles per expert = ceil(Tk_e / BLOCK_M)
    m_tiles = (expert_counts + BLOCK_M - 1) // BLOCK_M
    tiles_per_expert = m_tiles * num_n_tiles
    tile_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    tile_offsets[1:] = torch.cumsum(tiles_per_expert, dim=0)
    return tile_offsets


def _route_and_permute_local_fused(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    E_global: int,
    E_local: int,
    TOP_K: int,
    N_GROUP: int,
    TOPK_GROUP: int,
):
    """Triton-backed fused routing + local permutation with downstream-compatible outputs."""
    T = int(routing_logits.shape[0])
    device = routing_logits.device

    group_size = E_global // N_GROUP
    expert_counts = torch.zeros((E_local,), dtype=torch.int32, device=device)

    # Route + reorder in a single kernel by writing to expert-major buckets.
    # Each local expert can receive at most one assignment per token, so bucket length T is safe.
    bucket_token_ids = torch.empty((E_local, T), dtype=torch.int32, device=device)
    bucket_weights = torch.empty((E_local, T), dtype=torch.float32, device=device)

    grid_route = (T,)
    _route_and_bucket_local_kernel[grid_route](
        routing_logits,
        routing_bias,
        expert_counts,
        bucket_token_ids,
        bucket_weights,
        T,
        stride_rl_t=routing_logits.stride(0), stride_rl_e=routing_logits.stride(1),
        stride_bucket_tok_e=bucket_token_ids.stride(0),
        stride_bucket_tok_t=bucket_token_ids.stride(1),
        stride_bucket_w_e=bucket_weights.stride(0),
        stride_bucket_w_t=bucket_weights.stride(1),
        local_start=int(local_expert_offset),
        routed_scaling_factor=float(routed_scaling_factor),
        E_GLOBAL=E_global,
        E_LOCAL=E_local,
        TOP_K=TOP_K,
        N_GROUP=N_GROUP,
        TOPK_GROUP=TOPK_GROUP,
        GROUP_SIZE=group_size,
        num_warps=8,
        num_stages=2,
    )

    total_tokens = int(expert_counts.sum().item())
    if total_tokens == 0:
        return None

    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    sorted_token_ids = torch.empty((total_tokens,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((total_tokens,), dtype=torch.float32, device=device)

    BLOCK_COMPACT_T = 128
    grid_compact = (E_local, triton.cdiv(T, BLOCK_COMPACT_T))
    _compact_expert_buckets_kernel[grid_compact](
        bucket_token_ids,
        bucket_weights,
        expert_counts,
        expert_offsets,
        sorted_token_ids,
        sorted_weights,
        stride_bucket_tok_e=bucket_token_ids.stride(0),
        stride_bucket_tok_t=bucket_token_ids.stride(1),
        stride_bucket_w_e=bucket_weights.stride(0),
        stride_bucket_w_t=bucket_weights.stride(1),
        E_LOCAL=E_local,
        BLOCK_T=BLOCK_COMPACT_T,
        num_warps=4,
    )

    return sorted_token_ids, sorted_weights, expert_counts, expert_offsets, total_tokens


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
    # Phase 1+2: Fused Routing + Token Permutation
    # ========================================================================
    route_outputs = _route_and_permute_local_fused(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        routed_scaling_factor=routed_scaling_factor,
        local_expert_offset=local_expert_offset,
        E_global=E_global,
        E_local=E_local,
        TOP_K=TOP_K,
        N_GROUP=N_GROUP,
        TOPK_GROUP=TOPK_GROUP,
    )

    if route_outputs is None:
        output.zero_()
        return

    sorted_token_ids, sorted_weights, expert_counts, expert_offsets, total_tokens = route_outputs

    permuted_input = hidden_states.index_select(0, sorted_token_ids.long()).contiguous()
    permuted_hs_scale = hidden_states_scale.index_select(1, sorted_token_ids.long()).contiguous()

    # ========================================================================
    # Phase 3: Persistent Grouped GEMM1
    # [total_tokens, H] x [E_local, 2*I, H]^T -> [total_tokens, 2*I]
    # Single kernel launch for all experts
    # ========================================================================
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    # Stage-wise precision switches for controlled A/B experiments.
    GEMM1_OUT_BF16 = False
    SWIGLU_OUT_BF16 = True
    GEMM2_OUT_BF16 = True
    # Default to TF32 compute path for numerical stability.
    GEMM2_USE_BF16_DOT = True

    gemm1_dtype = torch.bfloat16 if GEMM1_OUT_BF16 else torch.float32
    swiglu_dtype = torch.bfloat16 if SWIGLU_OUT_BF16 else torch.float32
    gemm2_dtype = torch.bfloat16 if GEMM2_OUT_BF16 else torch.float32

    gemm1_out = torch.empty((total_tokens, 2 * I), dtype=gemm1_dtype, device=device)

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
            STORE_BF16=GEMM1_OUT_BF16,
            num_warps=8, num_stages=3,
        )

    # ========================================================================
    # Phase 4: SwiGLU — [total_tokens, 4096] -> [total_tokens, 2048]
    # ========================================================================
    swiglu_out = torch.empty((total_tokens, I), dtype=swiglu_dtype, device=device)
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
        STORE_BF16=SWIGLU_OUT_BF16,
        num_warps=4,
    )

    # ========================================================================
    # Phase 5: Persistent Grouped GEMM2
    # [total_tokens, I] x [E_local, H, I]^T -> [total_tokens, H]
    # Single kernel launch for all experts
    # ========================================================================
    gemm2_out = torch.empty((total_tokens, H), dtype=gemm2_dtype, device=device)

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
            GEMM2_USE_BF16_DOT=GEMM2_USE_BF16_DOT,
            STORE_BF16=GEMM2_OUT_BF16,
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
        num_warps=8,
    )

    output.copy_(out_accum.to(torch.bfloat16))