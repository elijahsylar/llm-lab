"""Registry tests: no GPU, no Ollama, no model files needed."""
import os
import tempfile
import unittest

from backend import model_registry as mr


class SlugTests(unittest.TestCase):
    def test_strips_quant_suffix(self):
        self.assertEqual(mr._slug("Qwen2.5-0.5B-Instruct-Q8_0"), "qwen2-5-0-5b-instruct")

    def test_strips_f16(self):
        self.assertEqual(mr._slug("mistral-7b-base-f16"), "mistral-7b-base")


class FormatGuessTests(unittest.TestCase):
    def test_families(self):
        self.assertEqual(mr._ollama_format_guess("llama"), "llama3")
        self.assertEqual(mr._ollama_format_guess("qwen2"), "chatml")
        self.assertEqual(mr._ollama_format_guess("gemma3"), "gemma")
        self.assertEqual(mr._ollama_format_guess(""), mr.DEFAULT_FORMAT)


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, name, size=3):
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(b"x" * size)

    def test_registers_ggufs_and_skips_companions(self):
        self.touch("SmolLM2-1.7B-Instruct-Q8_0.gguf")
        self.touch("model-mmproj-f16.gguf")
        self.touch("notes.txt")
        reg = {}
        mr._scan_gguf_dir(self.dir, reg)
        self.assertEqual(list(reg), ["smollm2-1-7b-instruct"])
        entry = reg["smollm2-1-7b-instruct"]
        self.assertEqual(entry["provider"], "llama_cpp")
        self.assertEqual(entry["prompt_format"], "chatml")
        self.assertEqual(entry["size_bytes"], 3)

    def test_lite_exports_get_lab_scope_and_base_meta(self):
        self.touch("Qwen2.5-0.5B-Instruct--run1-Q8_0.gguf")
        reg = {}
        mr._scan_gguf_dir(self.dir, reg, scope="lab")
        (entry,) = reg.values()
        self.assertEqual(entry["scope"], "lab")
        self.assertEqual(entry["context_length"], 2048)
        self.assertIn("run1-Q8_0", entry["description"])

    def test_key_collision_gets_stable_suffix(self):
        self.touch("a-Q4_K_M.gguf")
        reg = {"a": {"key": "a"}}
        mr._scan_gguf_dir(self.dir, reg)
        reg2 = {"a": {"key": "a"}}
        mr._scan_gguf_dir(self.dir, reg2)
        new = [k for k in reg if k != "a"]
        self.assertEqual(len(new), 1)
        self.assertEqual(new, [k for k in reg2 if k != "a"])

    def test_missing_dir_is_ignored(self):
        reg = {}
        mr._scan_gguf_dir(os.path.join(self.dir, "nope"), reg)
        self.assertEqual(reg, {})

    def test_public_entry_hides_path(self):
        self.touch("SmolLM2-1.7B-Instruct-Q8_0.gguf")
        reg = {}
        mr._scan_gguf_dir(self.dir, reg)
        pub = mr.public_entry(next(iter(reg.values())))
        self.assertNotIn("path", pub)
        self.assertNotIn("scope", pub)


if __name__ == "__main__":
    unittest.main()
