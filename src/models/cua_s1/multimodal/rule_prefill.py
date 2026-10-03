"""Pinned Qwen3.5 cache-free prefill with an explicit DeltaNet rule callable.

Adapted from Transformers 5.17.0 modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward
(Copyright the HuggingFace Inc. team, Apache License 2.0). Operations retain the
upstream order and original token shapes; only the supplied rule may bucket.
No global functions, instance methods, weights or persistent states are changed.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def pinned_implementation():
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if transformers.__version__ != "5.17.0":
        raise RuntimeError("rule-bucket requires pinned Transformers 5.17.0")
    return modeling_qwen3_5


def rule_prefill(module, hidden, rule):
    """Dense, cache-free prefill on the supplied module's existing parameters."""
    import torch
    from torch.nn import functional as F

    implementation = pinned_implementation()
    batch, length, _ = hidden.shape
    mixed = module.in_proj_qkv(hidden).transpose(1, 2)
    z = module.in_proj_z(hidden).reshape(batch, length, -1, module.head_v_dim)
    b = module.in_proj_b(hidden)
    a = module.in_proj_a(hidden)
    mixed = implementation.causal_conv1d_fn(
        mixed,
        module.conv1d.weight.squeeze(1),
        module.conv1d.bias,
        activation=module.activation,
    ).transpose(1, 2)
    query, key, value = torch.split(
        mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1
    )
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    value = value.reshape(batch, length, -1, module.head_v_dim)
    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    repeats = module.num_v_heads // module.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    output, _ = rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    output = module.norm(
        output.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim)
    )
    return module.out_proj(output.reshape(batch, length, -1))
