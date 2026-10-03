"""Bounded CUDA Graph replay for Qwen3.5 multimodal linear-attention runs.

Full-attention SDPA changes BF16 results under whole-model Graph capture on the
pinned RTX 4090 stack. Keep those layers eager and capture the intervening
Gated DeltaNet runs, whose replay matches the original forward bit for bit.
"""

from __future__ import annotations

import ctypes
import gc
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field


def tensor_signature(value):
    if value is None:
        return None
    return (
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
    )


@dataclass(frozen=True)
class GraphConfig:
    max_shapes: int = 8
    max_bytes: int = 1024 * 1024 * 1024
    min_uses: int = 2
    max_tokens: int = 2048

    def __post_init__(self):
        if min(self.max_shapes, self.max_bytes, self.min_uses, self.max_tokens) <= 0:
            raise ValueError("graph limits must be positive")


class GraphCache:
    """LRU cache whose entries own all segments for one tensor layout."""

    def __init__(self, max_shapes: int, max_bytes: int):
        self.max_shapes = max_shapes
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0

    def __len__(self):
        return len(self.entries)

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        if value.bytes > self.max_bytes:
            return None
        removed = []
        if key in self.entries:
            previous = self.entries.pop(key)
            self.bytes -= previous.bytes
            removed.append(previous)
        self.entries[key] = value
        self.bytes += value.bytes
        while len(self.entries) > self.max_shapes or self.bytes > self.max_bytes:
            _, previous = self.entries.popitem(last=False)
            self.bytes -= previous.bytes
            removed.append(previous)
        return removed

    def clear(self):
        removed = list(self.entries.values())
        self.entries.clear()
        self.bytes = 0
        return removed


class _GraphPool:
    """One shape's private allocator pool and exclusively owned capture stream.

    Segments share scratch space only within a shape, in capture/replay order.
    All graphs must be reset before this stream is destroyed. Other shapes use
    separate streams because the pinned PyTorch clears cuBLAS workspaces by
    capture stream when destroying a graph.
    """

    def __init__(self, device):
        import torch

        self._torch = torch
        self.device = device
        self._stream_ptr = None
        if torch.cuda.get_allocator_backend() != "native":
            raise RuntimeError(
                "graph memory accounting requires the native CUDA allocator"
            )
        self.pool_id = torch.cuda.graph_pool_handle()
        with torch.cuda.device(device):
            pointer = ctypes.c_void_p()
            torch.cuda.check_error(
                torch.cuda.cudart().cudaStreamCreate(ctypes.addressof(pointer))
            )
            self._stream_ptr = pointer.value
            self.capture_stream = torch.cuda.ExternalStream(
                self._stream_ptr, device=device
            )

    def reserved_bytes(self):
        # Allocated tensor deltas omit inactive graph scratch space, which stays
        # reserved for replay. Query this pool directly, not process-wide deltas.
        return sum(
            segment["total_size"]
            for segment in self._torch.cuda.memory_snapshot(
                mempool_id=self.pool_id, include_traces=False
            )
        )

    def close(self):
        if getattr(self, "_stream_ptr", None) is None:
            return
        torch = self._torch
        with torch.cuda.device(self.device):
            torch.cuda.check_error(
                torch.cuda.cudart().cudaStreamDestroy(self._stream_ptr)
            )
            self._stream_ptr = None
            self.capture_stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 -- best-effort destructor cleanup
            pass


class _GraphSegment:
    """One fixed-layout capture of adjacent linear-attention decoder layers."""

    def __init__(self, layers, start, end, hidden, mask, pool=None):
        import torch

        self._torch = torch
        self._device = hidden.device
        self._pool = None
        self._owns_pool = pool is None
        self.graph = None
        self.capture_stream = None
        self.layers = layers
        self.start = start
        self.end = end
        self.input_signature = tensor_signature(hidden)
        self.mask_signature = tensor_signature(mask)
        self.static_hidden = hidden.clone()
        self.static_mask = mask.clone() if mask is not None else None
        self.external_bytes = self.static_hidden.untyped_storage().nbytes()
        if self.static_mask is not None:
            self.external_bytes += self.static_mask.untyped_storage().nbytes()

        stream = torch.cuda.Stream(device=hidden.device)
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream(hidden.device).wait_stream(stream)

        try:
            with torch.cuda.device(self._device):
                self._pool = pool if pool is not None else _GraphPool(self._device)
                self.capture_stream = self._pool.capture_stream
                self.graph = torch.cuda.CUDAGraph()
                with (
                    torch.cuda.graph(
                        self.graph,
                        stream=self.capture_stream,
                        pool=self._pool.pool_id,
                    ),
                    torch.no_grad(),
                ):
                    self.static_output = self._forward()
        except BaseException:
            # Preserve the actual capture failure if a poisoned CUDA context
            # also prevents cleanup. The destructor makes a best-effort retry.
            with suppress(Exception):
                self.close()
            raise

    def close(self):
        """Retire a graph; shared-pool segments must all retire together."""
        if getattr(self, "_pool", None) is None:
            return
        torch = self._torch
        with torch.cuda.device(self._device):
            # Replays run on the caller's stream, not the capture stream.
            torch.cuda.synchronize(self._device)
            if self.graph is not None:
                self.graph.reset()
                self.graph = None
            self.static_output = None
            self.static_hidden = None
            self.static_mask = None
            if self._owns_pool:
                self._pool.close()
            self._pool = None
            self.capture_stream = None

    def __del__(self):
        # Interpreter teardown and failed CUDA contexts cannot safely raise.
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 -- best-effort destructor cleanup
            pass

    def _forward(self):
        hidden = self.static_hidden
        for index in range(self.start, self.end):
            hidden = self.layers[index](
                hidden,
                position_embeddings=None,
                attention_mask=self.static_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
            )
        return hidden

    def replay(self, hidden, mask):
        if (
            tensor_signature(hidden) != self.input_signature
            or tensor_signature(mask) != self.mask_signature
        ):
            raise ValueError("CUDA Graph segment layout changed")
        self.static_hidden.copy_(hidden)
        if self.static_mask is not None:
            self.static_mask.copy_(mask)
        self.graph.replay()
        return self.static_output


@dataclass
class _ShapeEntry:
    blocks: dict = field(default_factory=dict)
    bytes: int = 0
    pool: _GraphPool | None = None

    def update_bytes(self):
        self.bytes = self.pool.reserved_bytes() + sum(
            block.external_bytes for block in self.blocks.values()
        )

    def close(self):
        # No surviving segment may replay after the first reset clears the
        # shared stream's workspace. The runtime retires an entire shape at once.
        for block in self.blocks.values():
            block.close()
        self.blocks.clear()
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 -- best-effort destructor cleanup
            pass


class GraphRuntime:
    """Exact-length graph cache for a loaded, immutable multimodal model."""

    def __init__(self, model, config: GraphConfig | None = None):
        self.model = model
        self.config = config or GraphConfig()
        self.cache = GraphCache(self.config.max_shapes, self.config.max_bytes)
        self.uses = OrderedDict()
        self.disabled = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {
            "eager": 0,
            "unsupported": 0,
            "warmup": 0,
            "disabled": 0,
            "replays": 0,
            "captures": 0,
            "capture_ms": 0.0,
            "evictions": 0,
            "rejected": 0,
            "memory_budget": 0,
            "capture_oom": 0,
            "capture_error": 0,
            "numerical_mismatch": 0,
        }

    def invalidate(self):
        """Call after replacing model weights or adapters."""
        with self.lock:
            retired = self.cache.clear()
            self.uses.clear()
            self.disabled.clear()
            for entry in retired:
                entry.close()
        del retired
        gc.collect()

    def _eager(self, values):
        self.stats["eager"] += 1
        return self.model(**values, logits_to_keep=1, use_cache=False).logits[0, -1, :]

    def _supported(self, values):
        hidden = values["inputs_embeds"]
        if (
            self.model.training
            or hidden.device.type != "cuda"
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] > self.config.max_tokens
            or values["position_ids"].shape != (3, 1, hidden.shape[1])
        ):
            return False
        core = self.model.get_base_model().model
        text = core.language_model
        types = text.config.layer_types
        return len(types) == text.config.num_hidden_layers == len(text.layers) and set(
            types
        ) == {"linear_attention", "full_attention"}

    def _run_segments(self, values, entry):
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        core = self.model.get_base_model().model
        text = core.language_model
        hidden = values["inputs_embeds"]
        kwargs = {
            "config": text.config,
            "inputs_embeds": hidden,
            "attention_mask": values["attention_mask"],
            "past_key_values": None,
            "position_ids": None,
        }
        masks = {
            "full_attention": create_causal_mask(**kwargs),
            "linear_attention": create_recurrent_attention_mask(**kwargs),
        }
        rope = text.rotary_emb(hidden, values["position_ids"])
        index = 0
        while index < text.config.num_hidden_layers:
            kind = text.config.layer_types[index]
            if kind == "full_attention":
                hidden = text.layers[index](
                    hidden,
                    position_embeddings=rope,
                    attention_mask=masks[kind],
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                )
                index += 1
                continue
            end = index + 1
            while (
                end < text.config.num_hidden_layers
                and text.config.layer_types[end] == "linear_attention"
            ):
                end += 1
            block = entry.blocks.get((index, end))
            if block is None:
                if entry.pool is None:
                    entry.pool = _GraphPool(hidden.device)
                block = _GraphSegment(
                    text.layers,
                    index,
                    end,
                    hidden,
                    masks["linear_attention"],
                    pool=entry.pool,
                )
                entry.blocks[(index, end)] = block
                entry.update_bytes()
            hidden = block.replay(hidden, masks["linear_attention"])
            index = end
        hidden = text.norm(hidden)
        return self.model.get_base_model().lm_head(hidden[:, -1:, :])[0, -1, :]

    def forward(self, values):
        """Return fresh logits; no static graph output escapes the runtime lock."""
        import torch

        with self.lock, torch.no_grad():
            if not self._supported(values):
                self.stats["unsupported"] += 1
                return self._eager(values)
            key = (
                id(self.model),
                self.model.get_base_model().model.language_model.config._attn_implementation,
                *(
                    tensor_signature(values[name])
                    for name in ("inputs_embeds", "position_ids", "attention_mask")
                ),
            )
            if key in self.disabled:
                self.stats["disabled"] += 1
                return self._eager(values)
            entry = self.cache.get(key)
            if entry is not None:
                output = self._run_segments(values, entry)
                self.stats["replays"] += 1
                return output

            uses = self.uses.get(key, 0) + 1
            self.uses[key] = uses
            self.uses.move_to_end(key)
            while len(self.uses) > 128:
                self.uses.popitem(last=False)
            if uses < self.config.min_uses:
                self.stats["warmup"] += 1
                return self._eager(values)

            reference = self._eager(values)
            candidate = _ShapeEntry()
            try:
                torch.cuda.synchronize(values["inputs_embeds"].device)
                capture_start = time.perf_counter()
                output = self._run_segments(values, candidate)
                torch.cuda.synchronize(values["inputs_embeds"].device)
                capture_ms = (time.perf_counter() - capture_start) * 1000
            except torch.cuda.OutOfMemoryError:
                self.stats["capture_oom"] += 1
                candidate.close()
                del candidate
                gc.collect()
                torch.cuda.empty_cache()
                self._disable(key)
                return reference
            except RuntimeError:
                # Only a healthy CUDA context can continue serving eagerly.
                torch.cuda.synchronize(values["inputs_embeds"].device)
                self.stats["capture_error"] += 1
                candidate.close()
                del candidate
                self._disable(key)
                gc.collect()
                return reference

            if not torch.equal(reference, output):
                self.stats["numerical_mismatch"] += 1
                candidate.close()
                del candidate, output
                self._disable(key)
                gc.collect()
                return reference
            retired = self.cache.put(key, candidate)
            if retired is None:
                self.stats["memory_budget"] += 1
                candidate.close()
                del candidate
                self._disable(key)
                gc.collect()
                return reference
            self.stats["captures"] += 1
            self.stats["capture_ms"] += capture_ms
            self.stats["evictions"] += len(retired)
            if retired:
                for entry in retired:
                    entry.close()
                del entry
                del retired
                gc.collect()
            else:
                del retired
            return output

    def _disable(self, key):
        self.stats["rejected"] += 1
        self.disabled[key] = None
        while len(self.disabled) > 128:
            self.disabled.popitem(last=False)
