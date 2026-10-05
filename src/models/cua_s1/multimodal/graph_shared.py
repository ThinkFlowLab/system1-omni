"""Per-model shared resource ownership for exact and rule-only CUDA Graphs.

Children are owned exclusively by this group; use its request and lifecycle
methods. Cache limits cover resident ownership, not transient candidate capture
or the model and other process allocations.
"""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from dataclasses import replace

from .graph_admission import AdmissionPolicy
from .graph_buckets import RuleBucketRuntime
from .graph_runtime import GraphCache, GraphConfig, GraphRuntime


class _CacheView:
    def __init__(self, group, mode):
        self.group = group
        self.mode = mode

    def get(self, key):
        return self.group.cache.get((self.mode, key))

    def put(self, key, entry):
        entry.key = (self.mode, key)
        retired = self.group.cache.put(entry.key, entry)
        if retired is not None:
            # Explicitly release even when diagnostics retain Python references.
            # _capture subsequently applies global cooldown to these exact keys.
            for previous in retired:
                previous.close()
        return retired


class _AdmissionView:
    def __init__(self, group, mode):
        self.group = group
        self.mode = mode

    def reason(self, key):
        return self.group.admission.reason((self.mode, key))

    def start_capture(self):
        return self.group.admission.start_capture()

    def finish_capture(self, ticket, elapsed_ms):
        self.group.admission.finish_capture(ticket, elapsed_ms)

    def evict(self, namespaced_key):
        self.group.admission.evict(namespaced_key)


class SharedGraphRuntime:
    """One request clock, capture ledger, LRU and lock for two execution modes."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or GraphConfig()
        self.cache = GraphCache(self.config.max_shapes, self.config.max_bytes)
        self.admission = AdmissionPolicy(self.config)
        self.lock = threading.RLock()
        self._in_request = False
        self._closed = False
        self._requests = 0
        self._mode = "eager"
        self._selected_this_request = set()
        self._runtimes = {}
        for mode, runtime_type in (
            ("exact", GraphRuntime),
            ("rule-bucket", RuleBucketRuntime),
        ):
            config = replace(
                self.config,
                mode=mode,
                bucket_width=64 if mode == "exact" else self.config.bucket_width,
            )
            self._runtimes[mode] = runtime_type(
                model,
                config,
                cache=_CacheView(self, mode),
                admission=_AdmissionView(self, mode),
                lock=self.lock,
            )

    @property
    def stats(self):
        result = {}
        for runtime in self._runtimes.values():
            for key, value in runtime.stats.items():
                result[key] = result.get(key, 0) + value
        result["requests"] = self._requests
        return result

    def select_mode(self, mode):
        if mode not in {"eager", "exact", "rule-bucket"}:
            raise ValueError("mode must be eager, exact or rule-bucket")
        with self.lock:
            if self._closed:
                raise RuntimeError("shared Graph runtime is closed")
            if self._in_request:
                raise RuntimeError("select a mode before beginning a request")
            self._mode = mode

    @contextmanager
    def request(self):
        with self.lock:
            if self._closed:
                raise RuntimeError("shared Graph runtime is closed")
            if self._in_request:
                raise RuntimeError("nested Graph requests are unsupported")
            self._in_request = True
            self._requests += 1
            self.admission.begin_request()
            self._selected_this_request.clear()
            for runtime in self._runtimes.values():
                runtime._in_request = True
            try:
                yield
            finally:
                for runtime in self._runtimes.values():
                    runtime._in_request = False
                self._in_request = False

    def forward(self, values):
        import torch

        with self.lock, torch.no_grad():
            if self._closed:
                raise RuntimeError("shared Graph runtime is closed")
            runtime = self._runtimes["exact" if self._mode == "eager" else self._mode]
            if self._in_request and self._mode not in self._selected_this_request:
                runtime.stats["requests"] += 1
                self._selected_this_request.add(self._mode)
            if self._mode == "eager":
                if not self._in_request:
                    runtime.stats["no_request"] += 1
                return runtime._eager(values)
            return runtime.forward(values)

    def invalidate(self):
        with self.lock:
            if self._in_request:
                raise RuntimeError("cannot invalidate during a Graph request")
            retired = self.cache.clear()
            for entry in retired:
                entry.close()
            self.admission.reset()
            for runtime in self._runtimes.values():
                runtime.disabled.clear()
        gc.collect()

    def close(self):
        with self.lock:
            if self._closed:
                return
            self.invalidate()
            for runtime in self._runtimes.values():
                runtime._closed = True
            self._closed = True
