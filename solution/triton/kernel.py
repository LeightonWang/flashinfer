import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe


NUM_EXPERTS_GLOBAL = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = 128


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def _get_tile_tokens_dim(seq_len: int, top_k: int, num_experts: int) -> int:
    tokens_per_expert = (seq_len * top_k) // num_experts
    tile_tokens_dim = _next_power_of_2(tokens_per_expert)
    tile_tokens_dim = max(tile_tokens_dim, 8)
    tile_tokens_dim = min(tile_tokens_dim, 64)
    return tile_tokens_dim


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
):
    seq_len, num_experts = routing_logits.shape
    local_num_experts = gemm1_weights.shape[0]

    assert num_experts == NUM_EXPERTS_GLOBAL
    assert hidden_states.shape == (seq_len, HIDDEN_SIZE)
    assert hidden_states_scale.shape == (HIDDEN_SIZE // BLOCK_SIZE, seq_len)
    assert gemm1_weights.shape == (local_num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert gemm1_weights_scale.shape == (
        local_num_experts,
        (2 * INTERMEDIATE_SIZE) // BLOCK_SIZE,
        HIDDEN_SIZE // BLOCK_SIZE,
    )
    assert gemm2_weights.shape == (local_num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    assert gemm2_weights_scale.shape == (
        local_num_experts,
        HIDDEN_SIZE // BLOCK_SIZE,
        INTERMEDIATE_SIZE // BLOCK_SIZE,
    )
    assert routing_bias is None or routing_bias.shape[-1] == NUM_EXPERTS_GLOBAL

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    else:
        local_expert_offset = int(local_expert_offset)

    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    tile_tokens_dim = _get_tile_tokens_dim(seq_len, TOP_K, NUM_EXPERTS_GLOBAL)

    routing_logits_f32 = routing_logits.to(torch.float32).contiguous()
    hidden_states_scale_f32 = hidden_states_scale.to(torch.float32).contiguous()
    gemm1_weights_scale_f32 = gemm1_weights_scale.to(torch.float32).contiguous()
    gemm2_weights_scale_f32 = gemm2_weights_scale.to(torch.float32).contiguous()

    if routing_bias is not None:
        routing_bias = routing_bias.contiguous()

    return trtllm_fp8_block_scale_moe(
        routing_logits_f32,
        routing_bias,
        hidden_states.contiguous(),
        hidden_states_scale_f32,
        gemm1_weights.contiguous(),
        gemm1_weights_scale_f32,
        gemm2_weights.contiguous(),
        gemm2_weights_scale_f32,
        NUM_EXPERTS_GLOBAL,
        TOP_K,
        N_GROUP,
        TOPK_GROUP,
        INTERMEDIATE_SIZE,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        tile_tokens_dim=tile_tokens_dim,
        routing_method_type=2,
        use_shuffled_weight=False,
    )
