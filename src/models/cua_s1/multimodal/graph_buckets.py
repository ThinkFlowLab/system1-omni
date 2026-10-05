"""Opt-in rule-only CUDA Graph buckets for the pinned multimodal worker."""

from __future__ import annotations

import time

from .graph_runtime import (
    GraphConfig,
    GraphRuntime,
    _GraphPool,
    _GraphSegment,
    tensor_signature,
)
from .rule_prefill import pinned_implementation, rule_prefill


def bucket_length(length, width):
    return ((length + width - 1) // width) * width


class RuleBucketRuntime(GraphRuntime):
    """Bounded cache of rule graphs, with explicit model-instance dispatch."""

    def __init__(self, model, config=None):
        config = config or GraphConfig(mode="rule-bucket")
        super().__init__(model, config)
        self.width = config.bucket_width
        self.stats.update(
            length_checks=0,
            length_rejections=0,
            length_disabled=0,
            length_check_ms=0.0,
            real_tokens=0,
            padded_tokens=0,
        )

    @staticmethod
    def _dense(values):
        hidden = values["inputs_embeds"]
        mask = values["attention_mask"]
        return (
            hidden.is_contiguous()
            and mask is not None
            and tuple(mask.shape) == tuple(hidden.shape[:2])
            and bool((mask == 1).all())
        )

    def _key(self, values):
        hidden = values["inputs_embeds"]
        return (
            id(self.model),
            self.model.get_base_model().model.language_model.config._attn_implementation,
            bucket_length(hidden.shape[1], self.width),
            hidden.shape[2],
            str(hidden.dtype),
            str(hidden.device),
        )

    def _validate_replay(self, values, entry, output):
        import torch

        length = values["inputs_embeds"].shape[1]
        if length in entry.verified_lengths:
            return output
        started = time.perf_counter()
        reference = self._eager(values)
        self.stats["length_checks"] += 1
        if torch.equal(reference, output):
            entry.verified_lengths.add(length)
        else:
            entry.rejected_lengths.add(length)
            self.stats["length_rejections"] += 1
            output = reference
        self.stats["length_check_ms"] += (time.perf_counter() - started) * 1000
        return output

    def forward(self, values):
        import torch

        with self.lock, torch.no_grad():
            if self._closed:
                raise RuntimeError("Graph runtime is closed")
            if not self._in_request:
                self.stats["no_request"] += 1
                return self._eager(values)
            if not self._supported(values) or not self._dense(values):
                self.stats["unsupported"] += 1
                return self._eager(values)
            length = values["inputs_embeds"].shape[1]
            if bucket_length(length, self.width) > self.config.max_tokens:
                self.stats["unsupported"] += 1
                return self._eager(values)
            key = self._key(values)
            if key in self.disabled:
                self.stats["disabled"] += 1
                return self._eager(values)
            entry = self.cache.get(key)
            if entry is not None:
                if length in entry.rejected_lengths:
                    self.stats["length_disabled"] += 1
                    return self._eager(values)
                output = self._run_segments(values, entry)
                self.stats["replays"] += 1
                return self._validate_replay(values, entry, output)
            reason = self.admission.reason(key)
            if reason is not None:
                self.stats[reason] += 1
                return self._eager(values)
            ticket = self.admission.start_capture()
            self.stats["capture_attempts"] += 1
            attempt_start = time.perf_counter()
            try:
                output = self._capture(values, key)
                entry = self.cache.get(key)
                if entry is not None:
                    entry.verified_lengths = {length}
                    entry.rejected_lengths = set()
                return output
            finally:
                elapsed = (time.perf_counter() - attempt_start) * 1000
                self.admission.finish_capture(ticket, elapsed)
                self.stats["capture_attempt_ms"] += elapsed

    def _run_segments(self, values, entry):
        from transformers.masking_utils import create_causal_mask

        implementation = pinned_implementation()
        text = self.model.get_base_model().model.language_model
        hidden = values["inputs_embeds"]
        length = hidden.shape[1]
        self.stats["real_tokens"] += length
        self.stats["padded_tokens"] += bucket_length(length, self.width) - length
        mask = create_causal_mask(
            config=text.config,
            inputs_embeds=hidden,
            attention_mask=values["attention_mask"],
            past_key_values=None,
            position_ids=None,
        )
        rope = text.rotary_emb(hidden, values["position_ids"])
        for index, layer in enumerate(text.layers):
            if text.config.layer_types[index] == "full_attention":
                hidden = layer(
                    hidden,
                    position_embeddings=rope,
                    attention_mask=mask,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                )
                continue
            if type(layer.linear_attn) is not implementation.Qwen3_5GatedDeltaNet:
                raise RuntimeError("unsupported linear-attention implementation")

            def dispatch(query, key, value, *, g, beta, **kwargs):
                packed = pack_rule_inputs(
                    dict(query=query, key=key, value=value, g=g, beta=beta), self.width
                )
                block = entry.blocks.get(index)
                if block is None:
                    if entry.pool is None:
                        entry.pool = _GraphPool(query.device)
                    block = RuleSegment(
                        implementation.torch_chunk_gated_delta_rule, packed, entry.pool
                    )
                    entry.blocks[index] = block
                    entry.update_bytes()
                output, state = block.replay_values(packed)
                return output[:, :length].contiguous(), state

            residual = hidden
            hidden = rule_prefill(
                layer.linear_attn, layer.input_layernorm(hidden), dispatch
            )
            hidden = residual + hidden
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = text.norm(hidden)
        return self.model.get_base_model().lm_head(hidden[:, -1:, :])[0, -1, :]


def pack_rule_inputs(values, width):
    import torch.nn.functional as F

    length = values["query"].shape[1]
    padding = bucket_length(length, width) - length
    return {
        # Zero-padding is a clone that can preserve noncontiguous strides.
        # Canonicalize even on an exact boundary so one bucket has one layout.
        name: F.pad(value, (0, 0) * (value.ndim - 2) + (0, padding)).contiguous()
        for name, value in values.items()
    }


class RuleSegment(_GraphSegment):
    """Capture just the fallback rule, reusing #33 stream/pool ownership."""

    def __init__(self, function, values, pool):
        self.function = function
        self.signatures = {k: tensor_signature(v) for k, v in values.items()}
        self.static_extra = {k: v.clone() for k, v in values.items() if k != "query"}
        super().__init__([], 0, 0, values["query"], None, pool=pool)
        self.external_bytes += sum(
            v.untyped_storage().nbytes() for v in self.static_extra.values()
        )

    def _forward(self):
        return self.function(
            self.static_hidden,
            self.static_extra["key"],
            self.static_extra["value"],
            g=self.static_extra["g"],
            beta=self.static_extra["beta"],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

    def replay_values(self, values):
        if {k: tensor_signature(v) for k, v in values.items()} != self.signatures:
            raise ValueError("CUDA Graph rule input layout changed")
        for name, value in self.static_extra.items():
            value.copy_(values[name])
        return super().replay(values["query"], None)

    def close(self):
        super().close()
        if hasattr(self, "static_extra"):
            self.static_extra.clear()
