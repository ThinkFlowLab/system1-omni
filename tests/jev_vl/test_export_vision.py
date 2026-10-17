"""Reject unsupported vision exports before loading weights or dependencies."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "jev_vl_export_vision", REPO / "recipe/jev_vl/export_vision.py"
)
export_vision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_vision)

VALID_CONFIG = json.loads((REPO / "tests/qwen3_5/data/config.json").read_text())

# The exporter compares the whole file content against this exact contract:
# wrong values, extra keys and missing keys must all reject.
EXPECTED_PROCESSOR = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}

LANGUAGE_ADAPTER = {
    "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.weight",
    "base_model.model.lm_head.lora_A.weight",
}
VISUAL_TENSORS = (
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.merger.linear_fc2.weight",
)
SHARD = "model-00001-of-00002.safetensors"


class _Finite:
    def __init__(self, value):
        self._value = value

    def all(self):
        return self._value


class _Weight:
    def __init__(self, dtype, finite=True):
        self.dtype = dtype
        self._finite = finite

    def contiguous(self):
        return self


class _FakeSafeOpen:
    def __init__(self, weights):
        self._weights = weights

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self._weights)

    def get_tensor(self, name):
        return self._weights[name]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportVisionTests(unittest.TestCase):
    def _fixtures(self, temporary, config=None, processor=None, index=None):
        model = Path(temporary) / "model"
        (model / "adapter_vllm").mkdir(parents=True)
        (model / "config.json").write_text(json.dumps(
            config if config is not None else VALID_CONFIG))
        (model / "preprocessor_config.json").write_text(json.dumps(
            processor if processor is not None else EXPECTED_PROCESSOR))
        if index is not None:
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": index}))
        (model / "adapter_vllm" / "adapter_config.json").write_text("{}")
        (model / "adapter_vllm" / "adapter_model.safetensors").write_bytes(b"adapter")
        return model, Path(temporary) / "export"

    def _mock_stack(self, bf16=object(), adapter=None, shard=None,
                    save_effect=None):
        torch = mock.Mock(name="torch")
        torch.bfloat16 = bf16
        torch.isfinite = lambda tensor: _Finite(tensor._finite)
        files = {}
        if adapter is not None:
            files["adapter_model.safetensors"] = adapter
        if shard is not None:
            files[SHARD] = shard
        read_weights = mock.Mock(side_effect=lambda path, framework:
                                 _FakeSafeOpen(files[Path(path).name]))
        save_weights = mock.Mock(side_effect=save_effect)
        return read_weights, save_weights, mock.patch.dict(sys.modules, {
            "torch": torch,
            "safetensors": mock.Mock(safe_open=read_weights),
            "safetensors.torch": mock.Mock(save_file=save_weights),
        })

    def test_rejects_wrong_dimensions_before_weights_and_output(self):
        read_weights = mock.Mock()
        save_weights = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, {
            "torch": None,
            "safetensors": mock.Mock(safe_open=read_weights),
            "safetensors.torch": mock.Mock(save_file=save_weights),
        }):
            model = Path(temporary) / "model"
            model.mkdir()
            out = Path(temporary) / "export"
            for section, field, value in (
                ("", "model_type", "qwen2_vl"),
                ("vision_config", "hidden_size", 1024),
                ("vision_config", "out_hidden_size", 2560),
                ("text_config", "hidden_size", 2560),
            ):
                with self.subTest(section=section or "toplevel", field=field):
                    config = copy.deepcopy(VALID_CONFIG)
                    if section:
                        config[section][field] = value
                    else:
                        config[field] = value
                    (model / "config.json").write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, "expected the JEV-27B-VL"):
                        export_vision.export(model, out)
                    read_weights.assert_not_called()
                    save_weights.assert_not_called()
                    self.assertFalse(out.exists())

    def test_rejects_existing_output_before_reading_the_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(temporary)
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "output already exists"):
                export_vision.export(model, out)

    def test_rejects_processor_contract_mismatch_before_weights(self):
        extra = dict(EXPECTED_PROCESSOR, do_convert_rgb=True)
        missing = {key: value for key, value in EXPECTED_PROCESSOR.items()
                   if key != "merge_size"}
        drifted = dict(EXPECTED_PROCESSOR, patch_size=14)
        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(temporary)
            for label, processor in (
                ("drifted value", drifted),
                ("extra key", extra),
                ("missing key", missing),
            ):
                with self.subTest(label=label):
                    (model / "preprocessor_config.json").write_text(
                        json.dumps(processor))
                    read_weights, save_weights, patched = self._mock_stack()
                    with patched, self.assertRaisesRegex(
                            ValueError, "unsupported image processor"):
                        export_vision.export(model, out)
                    read_weights.assert_not_called()
                    save_weights.assert_not_called()
                    self.assertFalse(out.exists())

    def test_rejects_non_language_adapter_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(temporary, index={
                name: SHARD for name in VISUAL_TENSORS})
            adapter = set(LANGUAGE_ADAPTER) | {
                "base_model.model.model.visual.patch_embed.proj.weight"}
            read_weights, save_weights, patched = self._mock_stack(
                adapter={name: mock.Mock() for name in adapter})
            with patched, self.assertRaisesRegex(
                    ValueError, "unsupported non-language adapter tensor"):
                export_vision.export(model, out)
            save_weights.assert_not_called()
            self.assertFalse(out.exists())

    def test_rejects_checkpoint_without_vision_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(temporary, index={
                "model.language_model.layers.0.self_attn.q_proj.weight": SHARD})
            read_weights, save_weights, patched = self._mock_stack(
                adapter={name: mock.Mock() for name in LANGUAGE_ADAPTER})
            with patched, self.assertRaisesRegex(
                    ValueError, "no vision tensors"):
                export_vision.export(model, out)
            save_weights.assert_not_called()
            self.assertFalse(out.exists())

    def test_rejects_shard_escaping_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(
                temporary, index={VISUAL_TENSORS[0]: "../escaped.safetensors"})
            read_weights, save_weights, patched = self._mock_stack(
                adapter={name: mock.Mock() for name in LANGUAGE_ADAPTER})
            with patched, self.assertRaisesRegex(
                    ValueError, "shard escapes model directory"):
                export_vision.export(model, out)
            save_weights.assert_not_called()
            self.assertFalse(out.exists())

    def test_rejects_non_bf16_or_non_finite_vision_weights(self):
        bf16 = object()
        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(temporary, index={
                name: SHARD for name in VISUAL_TENSORS})
            for label, weight in (
                ("wrong dtype", _Weight(object())),
                ("non-finite", _Weight(bf16, finite=False)),
            ):
                with self.subTest(label=label):
                    read_weights, save_weights, patched = self._mock_stack(
                        bf16=bf16,
                        adapter={name: mock.Mock() for name in LANGUAGE_ADAPTER},
                        shard={VISUAL_TENSORS[0]: weight,
                               VISUAL_TENSORS[1]: _Weight(bf16)})
                    with patched, self.assertRaisesRegex(
                            ValueError, "expected finite BF16 vision weights"):
                        export_vision.export(model, out)
                    save_weights.assert_not_called()
                    self.assertFalse(out.exists())

    def test_export_writes_manifest_pins_for_valid_checkpoint(self):
        bf16 = object()
        saved = {}

        def record(tensors, path):
            saved.update(tensors)
            path.write_bytes(b"vision")

        with tempfile.TemporaryDirectory() as temporary:
            model, out = self._fixtures(temporary, index={
                name: SHARD for name in VISUAL_TENSORS})
            read_weights, save_weights, patched = self._mock_stack(
                bf16=bf16,
                adapter={name: mock.Mock() for name in LANGUAGE_ADAPTER},
                shard={name: _Weight(bf16) for name in VISUAL_TENSORS},
                save_effect=record)
            with patched:
                export_vision.export(model, out)
            self.assertEqual(set(saved), set(VISUAL_TENSORS))
            for name in ("config.json", "preprocessor_config.json",
                         "vision.safetensors", "jev_vl_vision.json"):
                self.assertTrue((out / name).exists(), name)
            manifest = json.loads((out / "jev_vl_vision.json").read_text())
            self.assertEqual(manifest["format"], "jev-vl-vision/1")
            self.assertEqual(manifest["model_id"], "autotrust/JEV-27B-VL")
            self.assertEqual(manifest["files"], {
                name: _sha256(out / name)
                for name in ("config.json", "preprocessor_config.json",
                             "vision.safetensors")})
            self.assertEqual(manifest["pins"], {
                "model_index_sha256": _sha256(
                    model / "model.safetensors.index.json"),
                "adapter_config_sha256": _sha256(
                    model / "adapter_vllm" / "adapter_config.json"),
                "adapter_model_sha256": _sha256(
                    model / "adapter_vllm" / "adapter_model.safetensors"),
            })


if __name__ == "__main__":
    unittest.main()
