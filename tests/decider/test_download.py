import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).parents[2] / "recipe/decider/download_weights.py"
spec = importlib.util.spec_from_file_location("decider_download", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Verification(unittest.TestCase):
    def test_rejects_content_and_size_mismatches(self):
        expected = (3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model"
            path.write_bytes(b"abc")
            module.verify(path, expected)
            path.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.verify(path, expected)
            path.write_bytes(b"abcd")
            with self.assertRaisesRegex(ValueError, "size"):
                module.verify(path, expected)


if __name__ == "__main__":
    unittest.main()
