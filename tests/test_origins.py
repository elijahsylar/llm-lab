"""Browser-origin tests: the CORS list, the PNA preflight and the served page.

They import backend.main with FastAPI's TestClient. No model is loaded and no
route that talks to the GPU or Ollama is called; llama_cpp is stubbed when it
is not installed.
"""
import importlib
import os
import sys
import types
import unittest
from unittest import mock

try:
    import llama_cpp  # noqa: F401
except ImportError:
    sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=object)

from fastapi.testclient import TestClient

ENV_VAR = "LLM_LAB_ALLOWED_ORIGINS"


def load_client(extra_origins=None):
    """Import backend.main afresh with ENV_VAR set (or unset) and wrap its app."""
    env = dict(os.environ)
    env.pop(ENV_VAR, None)
    if extra_origins is not None:
        env[ENV_VAR] = extra_origins
    with mock.patch.dict(os.environ, env, clear=True):
        import backend.main
        module = importlib.reload(backend.main)
    return TestClient(module.app)


def preflight(client, origin, pna=False):
    headers = {"Origin": origin, "Access-Control-Request-Method": "POST"}
    if pna:
        headers["Access-Control-Request-Private-Network"] = "true"
    return client.options("/api/chat", headers=headers)


class DefaultOriginTests(unittest.TestCase):
    def setUp(self):
        self.client = load_client()

    def test_null_origin_preflight_refused_by_default(self):
        r = preflight(self.client, "null")
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_null_origin_pna_preflight_refused_by_default(self):
        r = preflight(self.client, "null", pna=True)
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("access-control-allow-origin", r.headers)
        self.assertNotIn("access-control-allow-private-network", r.headers)

    def test_null_origin_gets_no_cors_header_on_simple_request(self):
        r = self.client.get("/", headers={"Origin": "null"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_unlisted_origin_preflight_refused(self):
        r = preflight(self.client, "https://evil.example.com")
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_allowed_origin_preflight_still_works(self):
        r = preflight(self.client, "http://localhost:8000")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "http://localhost:8000")

    def test_allowed_origin_pna_preflight_still_works(self):
        r = preflight(self.client, "http://localhost:5173", pna=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "http://localhost:5173")
        self.assertEqual(r.headers["access-control-allow-private-network"], "true")

    def test_known_limitation_unlisted_origin_can_fire_bodyless_stop_routes(self):
        # Documents current behaviour, not a goal: CORS is not access control.
        # A POST with no body needs no preflight, so the handler runs for any
        # origin; only reading the response is blocked. A JSON body sent as
        # text/plain (also preflight-free) is rejected before the handler.
        evil = {"Origin": "https://evil.example.com"}
        for path in ("/api/finetune/stop", "/api/tiny/stop"):
            r = self.client.post(path, headers=evil)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {"status": "not_running"})
            self.assertNotIn("access-control-allow-origin", r.headers)
        r = self.client.post(
            "/api/tiny/sample", headers={**evil, "Content-Type": "text/plain"},
            content='{"run_name": "no-such-run"}')
        self.assertEqual(r.status_code, 422)

    def test_frontend_served_at_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("const API", r.text)


class OptInOriginTests(unittest.TestCase):
    def setUp(self):
        self.client = load_client("https://pht.example.com, null")
        self.addCleanup(load_client)  # leave the module on its defaults

    def test_null_origin_opt_in_through_env(self):
        r = preflight(self.client, "null")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "null")

    def test_null_origin_opt_in_covers_pna_preflight(self):
        r = preflight(self.client, "null", pna=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "null")
        self.assertEqual(r.headers["access-control-allow-private-network"], "true")

    def test_env_origin_allowed_next_to_null(self):
        r = preflight(self.client, "https://pht.example.com")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "https://pht.example.com")


if __name__ == "__main__":
    unittest.main()
