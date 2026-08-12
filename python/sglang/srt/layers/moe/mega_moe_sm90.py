# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SM90 FP8 Mega-MoE forward path and expert-weight prep."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.utils import _device_sm

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


def is_sm90_fp8_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return (
        hasattr(deep_gemm, "fp8_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and getattr(experts, "_mega_moe_sm90_fp8_weights", False)
    )


def is_sm90_fp4_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return (
        hasattr(deep_gemm, "fp8_fp4_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and getattr(experts, "_mega_moe_sm90_fp4_weights", False)
    )


def get_sm90_fused_shared_weights(
    moe: DeepseekV2MoE,
    num_tokens: int,
) -> Optional[tuple]:
    """Return cached transformed weights for the decode shared phase.

    This path is deliberately opt-in while its accuracy/performance envelope
    is being validated.  Unsupported shapes and weight formats fall back to
    the existing standalone shared MLP without changing routed MegaMoE.
    """
    if os.getenv("SGLANG_MEGA_MOE_FUSE_SHARED_EXPERT", "0") != "1":
        return None
    max_tokens = int(os.getenv("SGLANG_MEGA_MOE_FUSE_SHARED_MAX_TOKENS", "64"))
    if (
        _device_sm != 90
        or num_tokens <= 0
        or num_tokens > max_tokens
        or getattr(moe.config, "n_shared_experts", 0) != 1
        or moe.num_fused_shared_experts != 0
        or not hasattr(moe, "shared_experts")
        or not getattr(moe, "_shared_expert_tp1", False)
        or not getattr(moe, "shared_experts_is_fp8", False)
        or getattr(moe.experts, "_mega_moe_sm90_fp4_weights", False)
        or not getattr(moe.experts, "_mega_moe_sm90_fp8_weights", False)
    ):
        return None

    import deep_gemm

    if not hasattr(deep_gemm, "fp8_mega_moe_with_shared"):
        return None
    cached = getattr(moe, "_mega_moe_sm90_shared_weights", None)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None

    shared = moe.shared_experts
    w13 = shared.gate_up_proj.weight.data
    w2 = shared.down_proj.weight.data
    w13_sf_raw = shared.gate_up_proj.weight_scale_inv.data
    w2_sf_raw = shared.down_proj.weight_scale_inv.data
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return None

    # Shared TP1 linears are 2-D; give the existing MegaMoE transform a
    # singleton expert dimension so their layout exactly matches routed FP8.
    w13_grouped = w13.unsqueeze(0)
    w2_grouped = w2.unsqueeze(0)
    w13_sf_grouped = (
        w13_sf_raw.unsqueeze(0) if w13_sf_raw.ndim == 2 else w13_sf_raw
    )
    w2_sf_grouped = (
        w2_sf_raw.unsqueeze(0) if w2_sf_raw.ndim == 2 else w2_sf_raw
    )
    _, n1, k1 = w13_grouped.shape
    _, n2, k2 = w2_grouped.shape
    w13_sf = deep_gemm.transform_sf_into_required_layout(
        w13_sf_grouped,
        mn=n1,
        k=k1,
        recipe=(128, 128),
        num_groups=1,
        disable_ue8m0_cast=True,
    )
    w2_sf = deep_gemm.transform_sf_into_required_layout(
        w2_sf_grouped,
        mn=n2,
        k=k2,
        recipe=(128, 128),
        num_groups=1,
        disable_ue8m0_cast=True,
    )
    cached = deep_gemm.transform_weights_for_mega_moe_sm90(
        (w13_grouped, w13_sf), (w2_grouped, w2_sf)
    )
    moe._mega_moe_sm90_shared_weights = cached
    return cached


def run_sm90_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
    shared_weights: Optional[tuple] = None,
) -> torch.Tensor:
    import deep_gemm

    # SM90 supports two weight recipes behind the same pre-dispatch:
    #   * FP8 weights  -> fp8_mega_moe        (per-128 FP32 SF)
    #   * packed FP4   -> fp8_fp4_mega_moe    (per-32 UE8M0 SFB, in-kernel decode)
    use_fp4 = getattr(moe.experts, "_mega_moe_sm90_fp4_weights", False)

    # Both SM90 paths feed FP8 activations with per-128 FP32 SF via
    # `mega_moe_pre_dispatch_sm90`. Enabling FP4 *activations* would allocate
    # buf.x as int8 (packed FP4) with a per-32 SF layout the SM90 kernels cannot
    # consume; the byte sizes can coincidentally match but the GEMM would read
    # the wrong scales and silently produce wrong results. Reject the combination.
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        raise RuntimeError(
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is incompatible with the "
            "SM90 mega-MoE paths (FP8 weights or FP4 weights). SM90 only supports "
            "FP8 activations with per-128 SF. Disable the flag or run on SM100."
        )

    if moe.experts.should_fuse_routed_scaling_factor_in_topk:
        routed_scaling_factor = 1.0
    else:
        routed_scaling_factor = float(moe.routed_scaling_factor)

    deep_gemm.mega_moe_pre_dispatch_sm90(
        hidden_states,
        topk_ids,
        topk_weights,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=128,
        routed_scaling_factor=routed_scaling_factor,
    )

    y = torch.empty(
        (max(num_tokens, 1), moe.config.hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    if use_fp4:
        assert shared_weights is None
        deep_gemm.fp8_fp4_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    elif shared_weights is not None:
        shared_l1_weights, shared_l2_weights = shared_weights
        deep_gemm.fp8_mega_moe_with_shared(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            shared_l1_weights,
            shared_l2_weights,
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    else:
        deep_gemm.fp8_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    y = y[:num_tokens]

    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)


def build_sm90_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    assert w13.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    num_groups, n1, k1 = w13.shape
    _, n2, k2 = w2.shape
    scale_group_mn, scale_group_k = 128, 128

    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid SM90 mega-moe K/group_size: k1={k1}, k2={k2}, "
        f"group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        w13_interleaved = _interleave_l1_weight_only(w13)
        experts.w13_weight.data = w13_interleaved
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.data,
        )
    else:
        import deep_gemm

        w13_sf = deep_gemm.transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=(128, 128),
            num_groups=num_groups,
            disable_ue8m0_cast=True,
        )
        w2_sf = deep_gemm.transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=(128, 128),
            num_groups=num_groups,
            disable_ue8m0_cast=True,
        )
        l1_pair, l2_pair = deep_gemm.transform_weights_for_mega_moe_sm90(
            (w13, w13_sf), (w2, w2_sf)
        )
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp8_weights = True
    experts._mega_moe_weights_built = True


def build_sm90_fp4_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    from deep_gemm import transform_weights_for_mega_moe_sm90_fp4

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    # SM90 FP4 weights ship as packed E2M1 (two nibbles per byte stored as
    # int8/uint8, last dim K//2) with raw per-32 FP32 SFB. The DeepGEMM helper
    # packs them into the interleaved FP4 + k-major UE8M0 layout the SM90 kernel
    # ldgs directly; no `transform_sf_into_required_layout` step is needed.
    assert w13.dtype in (torch.int8, torch.uint8)
    assert w2.dtype in (torch.int8, torch.uint8)

    l1_pair, l2_pair = transform_weights_for_mega_moe_sm90_fp4(
        (w13, w13_sf_fp32), (w2, w2_sf_fp32)
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Replace the checkpoint-layout params with the MegaMOE layout so the
        # originals can be released before KV-pool sizing. This drops the
        # non-MegaMOE fallback path for this layer (see should_use_mega_moe,
        # which raises instead of falling back once this flag is set).
        experts.w13_weight.data = l1_pair[0]
        experts.w2_weight.data = l2_pair[0]
        experts.w13_weight_scale_inv.data = l1_pair[1]
        experts.w2_weight_scale_inv.data = l2_pair[1]
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.data,
        )
    else:
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp4_weights = True
    experts._mega_moe_weights_built = True
