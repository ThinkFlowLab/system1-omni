"""Check image selection and recoverable asset writes without GPU dependencies."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "recipe/jev_vl/preencode.py"
SPEC = importlib.util.spec_from_file_location("jev_vl_preencode", SCRIPT)
preencode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preencode)


class ImageManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.jsonl"

    def images(self, *states):
        self.manifest.write_text("\n".join(
            json.dumps({"request": {"state": state}}) for state in states
        ) + "\n")
        return preencode.images_of(self.manifest)

    def test_recognizes_and_deduplicates_both_image_forms(self):
        first, second = "data:image/png;base64,AA==", "data:image/png;base64,AQ=="
        self.assertEqual(self.images(
            ["text", {"image": second}, {"type": "image_url", "image_url": {"url": first}}],
            [{"image": first}],
        ), [first, second])

    def test_structured_text_does_not_become_an_image(self):
        self.assertEqual(self.images(
            [{"image_url": "https://example.org/logo.png"},
             {"image_url": {"url": "https://example.org/logo.png"}},
             {"type": "text", "text": "Ready.", "image_url": {"url": "ignored"}},
             {"type": "metadata", "image_url": {"url": "ignored"}}],
            {"image": "top-level dictionaries are text"},
            "plain text", None,
        ), [])

    def test_image_shorthand_takes_precedence(self):
        self.assertEqual(self.images([{
            "type": "image_url", "image": "selected", "image_url": {"url": "ignored"},
        }]), ["selected"])
        for value in (None, "", False, 1, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "image part"):
                self.images([{"image": value, "type": "image_url", "image_url": {"url": "ignored"}}])

    def test_rejects_malformed_typed_image_parts(self):
        for image_url in (None, "bad", {}, {"url": None}, {"url": 1}, {"url": ""}):
            with self.subTest(image_url=image_url), self.assertRaisesRegex(ValueError, "image part"):
                self.images([{"type": "image_url", "image_url": image_url}])

    def test_help_and_text_only_manifest_need_no_model_dependencies(self):
        self.images([{"image_url": "ordinary metadata"}])
        commands = [
            ["--help"],
            ["--model", str(self.root / "missing-model"), "--manifest", str(self.manifest),
             "--out", str(self.root / "assets")],
        ]
        for args in commands:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-S", str(SCRIPT), *args],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 unique images", result.stdout)
        self.assertFalse((self.root / "assets").exists())


class AssetWriteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.url = "data:image/png;base64,AA=="
        self.key = preencode.hashlib.sha256(self.url.encode()).hexdigest()
        self.dest = self.root / "assets" / self.key
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps({"request": {"state": [{"image": self.url}]}}))
        (self.root / "model.safetensors.index.json").write_text("{}")

        tensor = mock.MagicMock(shape=(1, 5120))
        tensor.detach.return_value.cpu.return_value.to.return_value = tensor
        tensor.contiguous.return_value = tensor
        torch = mock.MagicMock(bfloat16="bfloat16")
        torch.prod.return_value.item.return_value = 4
        torch.isfinite.return_value.all.return_value = True
        model = mock.MagicMock(spatial_merge_size=2)
        model.return_value.pooler_output = tensor
        processor = mock.MagicMock()
        processor.return_value = {
            "image_grid_thw": [mock.Mock(tolist=lambda: [1, 2, 2])],
            "pixel_values": mock.Mock(),
        }
        transformers = mock.Mock()
        transformers.AutoProcessor.from_pretrained.return_value = processor
        self.save = mock.Mock(side_effect=lambda tensors, filename: Path(filename).write_bytes(b"rows"))
        for patcher in (
            mock.patch.dict(sys.modules, {
                "torch": torch, "safetensors": mock.Mock(),
                "safetensors.torch": mock.Mock(save_file=self.save), "transformers": transformers,
            }),
            mock.patch.object(preencode, "load_vision", return_value=model),
            mock.patch.object(preencode, "decode_url", return_value=object()),
            mock.patch.object(sys, "argv", [
                str(SCRIPT), "--model", str(self.root), "--manifest", str(self.manifest),
                "--out", str(self.root / "assets"),
            ]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_complete(self):
        self.assertEqual((self.dest / "emb.safetensors").read_bytes(), b"rows")
        grid = json.loads((self.dest / "grid.json").read_text())
        self.assertEqual(grid["url_sha256"], self.key)
        self.assertEqual(grid["grid_thw"], [1, 2, 2])
        self.assertEqual(grid["shape"], [1, 5120])

    def test_rebuilds_incomplete_assets(self):
        for existing in ((), ("emb.safetensors",), ("grid.json",)):
            with self.subTest(existing=existing):
                self.dest.mkdir(parents=True, exist_ok=True)
                for child in self.dest.iterdir():
                    child.unlink()
                for name in existing:
                    (self.dest / name).write_bytes(b"incomplete")
                self.save.reset_mock()
                preencode.main()
                self.save.assert_called_once()
                self.assert_complete()

    def test_retries_interrupted_metadata_write(self):
        original_write = Path.write_text

        def interrupted_write(path, text, *args, **kwargs):
            if path.parent == self.dest:
                original_write(path, text[:1], *args, **kwargs)
                raise OSError("interrupted metadata write")
            return original_write(path, text, *args, **kwargs)

        with mock.patch.object(Path, "write_text", interrupted_write):
            with self.assertRaisesRegex(OSError, "interrupted metadata write"):
                preencode.main()
        self.assertFalse((self.dest / "grid.json").exists())
        preencode.main()
        self.assertEqual(self.save.call_count, 2)
        self.assert_complete()

    def test_retries_failed_embedding_write_with_stale_grid(self):
        self.dest.mkdir(parents=True)
        (self.dest / "grid.json").write_text("{}")
        successful_save = self.save.side_effect

        def interrupted_save(tensors, filename):
            Path(filename).write_bytes(b"partial rows")
            raise OSError("interrupted embedding write")

        self.save.side_effect = interrupted_save
        with self.assertRaisesRegex(OSError, "interrupted embedding write"):
            preencode.main()
        self.assertFalse((self.dest / "grid.json").exists())
        self.save.side_effect = successful_save
        preencode.main()
        self.assert_complete()

    def test_preserves_complete_asset(self):
        preencode.main()
        expected = {p.name: p.read_bytes() for p in self.dest.iterdir()}
        self.save.reset_mock()
        preencode.main()
        self.save.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in self.dest.iterdir()}, expected)


if __name__ == "__main__":
    unittest.main()
