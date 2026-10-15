"""Tiny real safetensors exports: preserve BF16 base and raw FP32 adapter bytes."""
import errno
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("jemm_export", ROOT / "recipe/jemm/export.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class Tokenizer:
    def encode(self, label, **_kwargs):
        return [EXPORT.LABELS.index(label)]

    def apply_chat_template(self, messages, **_kwargs):
        return "system:" + messages[0]["content"] + "\n" + messages[1]["content"] + "\n<think>\n\n</think>"


def inventory(directory, names):
    files, tensors = {}, {}
    for name in names:
        files[name] = EXPORT.sha256(directory / name)
        if name.endswith(".safetensors"):
            with safe_open(str(directory / name), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    view = reader.get_slice(key)
                    tensors[key] = {"dtype": view.get_dtype(), "shape": view.get_shape(), "file": name}
    return {"files": files, "tensors": tensors}


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base, self.adapter, self.out = (self.root / name for name in ("base", "adapter", "out"))
        self.base.mkdir(); self.adapter.mkdir()
        self.target = "model.language_model.layers.0.mlp.down_proj.weight"
        self.language = {self.target: torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16),
                         "model.language_model.embed_tokens.weight": torch.ones(32, 2, dtype=torch.bfloat16)}
        save_file(self.language, str(self.base / "model-00001-of-00002.safetensors"))
        other = {"model.visual.patch_embed.weight": torch.ones(2, 2, dtype=torch.bfloat16),
                 "lm_head.weight": torch.arange(64).reshape(32, 2).to(torch.bfloat16),
                 "mtp.fc.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
        save_file(other, str(self.base / "model-00002-of-00002.safetensors"))
        mapping = {key: "model-00001-of-00002.safetensors" for key in self.language}
        mapping.update({key: "model-00002-of-00002.safetensors" for key in other})
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        (self.base / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 32, "hidden_size": 2}}))
        (self.base / "tokenizer.json").write_text("{}")
        (self.base / "preprocessor_config.json").write_text("{}")
        self.lora = {"base_model.model." + self.target.removesuffix(".weight") + ".lora_A.weight": torch.full((16, 2), 0.12345678),
                     "base_model.model." + self.target.removesuffix(".weight") + ".lora_B.weight": torch.full((2, 16), 0.23456789)}
        self.write_adapter()
        self.config = {"peft_type": "LORA", "base_model_name_or_path": EXPORT.BASE_MODEL_ID, "r": 16, "lora_alpha": 32}
        (self.adapter / "adapter_config.json").write_text(json.dumps(self.config))
        (self.adapter / "decision_config.json").write_text(json.dumps(EXPORT.CALIBRATION))
        self.refresh_inventory()

    def tearDown(self):
        self.temporary.cleanup()

    def write_adapter(self):
        save_file(self.lora, str(self.adapter / "adapter_model.safetensors"))

    def refresh_inventory(self):
        self.inventory = {"base": inventory(self.base, [p.name for p in self.base.iterdir()]),
                          "adapter": inventory(self.adapter, [p.name for p in self.adapter.iterdir()])}

    def run_export(self):
        return EXPORT.export_checkpoint(self.base, self.adapter, self.out, EXPORT.PINS,
                                        tokenizer=Tokenizer(), inventory=self.inventory)

    def test_preserves_original_base_and_fp32_adapter_without_merging(self):
        manifest = self.run_export()
        self.assertEqual(manifest["format"], "jemm-native/2")
        self.assertEqual(manifest["lora_mode"], "unmerged_fp32")
        self.assertEqual(manifest["lora_file"], "adapter_model.safetensors")
        self.assertEqual(manifest["lora_rank"], 16)
        self.assertEqual(manifest["lora_pairs"], 1)  # injected tiny inventory, not production
        self.assertEqual(manifest["lora_scale"], 2.)
        self.assertEqual(manifest["lora_sha256"], EXPORT.sha256(self.adapter / "adapter_model.safetensors"))
        mapping = json.loads((self.out / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(mapping, {key: "model-00001-of-00002.safetensors" for key in self.language})
        for name in set(mapping.values()):
            self.assertEqual((self.out / name).read_bytes(), (self.base / name).read_bytes())
        self.assertEqual((self.out / "adapter_model.safetensors").read_bytes(), (self.adapter / "adapter_model.safetensors").read_bytes())
        with safe_open(str(self.out / mapping[self.target]), framework="pt", device="cpu") as reader:
            self.assertTrue(torch.equal(reader.get_tensor(self.target), self.language[self.target]))
        with safe_open(str(self.out / "adapter_model.safetensors"), framework="pt", device="cpu") as reader:
            for key in self.lora:
                self.assertEqual(reader.get_tensor(key).dtype, torch.float32)
        with safe_open(str(self.out / "jemm_lm_head.safetensors"), framework="pt", device="cpu") as reader:
            self.assertTrue(torch.equal(reader.get_tensor("weight"), torch.arange(64).reshape(32, 2).to(torch.bfloat16)))
        with safe_open(str(self.out / "vision.safetensors"), framework="pt", device="cpu") as reader:
            self.assertTrue(torch.equal(reader.get_tensor("model.visual.patch_embed.weight"), torch.ones(2, 2, dtype=torch.bfloat16)))

    def test_cross_device_copy_fallback_preserves_identical_source_bytes(self):
        with patch.object(EXPORT.os, "link", side_effect=OSError(errno.EXDEV, "cross-device")):
            manifest = self.run_export()
        for name in ["model-00001-of-00002.safetensors", "adapter_model.safetensors"]:
            source = self.adapter if name.startswith("adapter") else self.base
            self.assertEqual((self.out / name).read_bytes(), (source / name).read_bytes())
            self.assertEqual(EXPORT.sha256(source / name), manifest["export_sha256"][name])

    def test_partial_export_resumes_and_writes_completion_marker_last(self):
        original = EXPORT.atomic_json
        def interrupted(path, value):
            original(path, value)
            if path.name == "export_progress.json" and len(value["completed"]) == 1:
                raise RuntimeError("simulated interruption")
        with patch.object(EXPORT, "atomic_json", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_export()
        self.assertFalse((self.out / "jemm_export.json").exists())
        original_inode = (self.out / "model-00001-of-00002.safetensors").stat().st_ino
        writes = []
        def recorded(path, value):
            writes.append(path.name); original(path, value)
        with patch.object(EXPORT, "atomic_json", side_effect=recorded):
            manifest = self.run_export()
        self.assertEqual(writes[-1], "jemm_export.json")
        self.assertEqual(original_inode, (self.out / "model-00001-of-00002.safetensors").stat().st_ino)
        self.assertEqual(self.run_export(), manifest)

    def test_nonfinite_adapter_is_refused_before_output_creation(self):
        next(iter(self.lora.values()))[0, 0] = float("nan")
        self.write_adapter(); self.refresh_inventory()
        with self.assertRaisesRegex(ValueError, "nonfinite adapter"):
            self.run_export()
        self.assertFalse(self.out.exists())

    def test_legacy_completed_export_requires_new_directory(self):
        self.run_export()
        path = self.out / "jemm_export.json"
        manifest = json.loads(path.read_text()); manifest["format"] = "jemm-native/1"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "legacy|jemm-native/2"):
            self.run_export()

    def test_pinned_inventory_is_rank16_and_has_exactly496_adapter_pairs(self):
        pinned = json.loads(EXPORT.INVENTORY_PATH.read_text())
        pairs, scale = EXPORT.adapter_pairs(pinned["base"]["tensors"], pinned["adapter"]["tensors"], self.config)
        self.assertEqual(len(pairs), 496); self.assertEqual(scale, 2.)
        self.assertEqual(len({v["file"] for key, v in pinned["base"]["tensors"].items()
                              if key.startswith("model.language_model.")}), 17)


if __name__ == "__main__":
    unittest.main()
