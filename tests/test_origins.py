"""Browser-origin tests: the CORS list, the PNA preflight, the served page and
the server-side refusal of state-changing requests from other sites' pages.

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

    def test_frontend_served_at_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("const API", r.text)


STOP_ROUTES = (("/api/finetune/stop", "lora"), ("/api/tiny/stop", "tiny"))
EVIL = "https://evil.example.com"


class StateChangeOriginTests(unittest.TestCase):
    """POST and other non-safe methods are refused by the server itself when
    they come from a page on an origin that is not allowed. CORS cannot do
    this: a POST with no body is sent without a preflight."""

    def setUp(self):
        self.client = load_client()
        self.main = sys.modules["backend.main"]

    def stop_with(self, path, kind, headers):
        """POST a stop route while a fake job of that kind holds the slot.
        Returns the response and whether the handler terminated the job."""
        proc = mock.Mock()
        with mock.patch.dict(self.main._job, {"proc": proc, "running": True, "kind": kind}):
            r = self.client.post(path, headers=headers)
        return r, proc.terminate.called

    def test_unlisted_origin_cannot_fire_stop_routes(self):
        for path, kind in STOP_ROUTES:
            r, terminated = self.stop_with(path, kind, {"Origin": EVIL})
            self.assertEqual(r.status_code, 403, path)
            self.assertFalse(terminated, path)

    def test_null_origin_cannot_fire_stop_routes_by_default(self):
        for path, kind in STOP_ROUTES:
            r, terminated = self.stop_with(path, kind, {"Origin": "null"})
            self.assertEqual(r.status_code, 403, path)
            self.assertFalse(terminated, path)

    def test_unlisted_origin_cannot_post_json_routes(self):
        r = self.client.post("/api/tiny/sample", headers={"Origin": EVIL},
                             json={"run_name": "no-such-run"})
        self.assertEqual(r.status_code, 403)

    def test_allowed_origin_can_fire_stop_routes(self):
        for path, kind in STOP_ROUTES:
            r, terminated = self.stop_with(path, kind, {"Origin": "http://localhost:5173"})
            self.assertEqual(r.status_code, 200, path)
            self.assertEqual(r.json(), {"status": "stopping", "kind": kind})
            self.assertTrue(terminated, path)

    def test_same_origin_post_works_on_any_loopback_host_and_port(self):
        for host in ("localhost:8000", "127.0.0.1:8000", "localhost:9000", "[::1]:8000"):
            r, terminated = self.stop_with(
                "/api/tiny/stop", "tiny", {"Host": host, "Origin": f"http://{host}"})
            self.assertEqual(r.status_code, 200, host)
            self.assertTrue(terminated, host)

    def test_origin_matching_a_non_loopback_host_is_refused(self):
        # DNS rebinding: the attacker's own name resolves to this machine, so
        # Origin and Host agree. Only loopback names count as same-origin.
        host = "evil.example.com:8000"
        r, terminated = self.stop_with(
            "/api/tiny/stop", "tiny", {"Host": host, "Origin": f"http://{host}"})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(terminated)

    def test_loopback_origin_on_another_port_is_not_same_origin(self):
        r, terminated = self.stop_with(
            "/api/tiny/stop", "tiny",
            {"Host": "localhost:8000", "Origin": "http://localhost:3000"})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(terminated)

    def test_post_without_origin_works(self):
        for path, kind in STOP_ROUTES:
            r, terminated = self.stop_with(path, kind, {})
            self.assertEqual(r.status_code, 200, path)
            self.assertTrue(terminated, path)

    def test_sec_fetch_site_cross_site_refused_without_origin(self):
        for path, kind in STOP_ROUTES:
            r, terminated = self.stop_with(path, kind, {"Sec-Fetch-Site": "cross-site"})
            self.assertEqual(r.status_code, 403, path)
            self.assertFalse(terminated, path)

    def test_sec_fetch_site_same_origin_works_without_origin(self):
        r, terminated = self.stop_with(
            "/api/tiny/stop", "tiny", {"Sec-Fetch-Site": "same-origin"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(terminated)

    def test_allowed_origin_works_when_browser_marks_it_cross_site(self):
        # A listed origin on another site (the https companion app) is
        # cross-site by definition; the origin list decides, not the hint.
        r, terminated = self.stop_with(
            "/api/tiny/stop", "tiny",
            {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(terminated)

    def test_get_routes_unaffected_by_origin(self):
        for headers in ({"Origin": EVIL}, {"Sec-Fetch-Site": "cross-site"}):
            r = self.client.get("/api/finetune/status", headers=headers)
            self.assertEqual(r.status_code, 200)
            self.assertIn("running", r.json())


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


    def test_null_origin_opt_in_covers_state_changing_requests(self):
        r = self.client.post("/api/tiny/stop", headers={"Origin": "null"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "not_running"})

    def test_env_origin_can_post(self):
        r = self.client.post("/api/tiny/stop", headers={
            "Origin": "https://pht.example.com", "Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
