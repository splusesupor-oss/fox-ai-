import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fox_ai_under_test", PROJECT / "fox_ai.py")
fox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fox)


def run_git(root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result.stdout.strip()


def init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    run_git(path, "init", "-b", "main")
    run_git(path, "config", "user.name", "Fox AI Test")
    run_git(path, "config", "user.email", "fox-ai-test@example.invalid")
    (path / "app.py").write_text("print('one')\n", encoding="utf-8")
    run_git(path, "add", "app.py")
    run_git(path, "commit", "-m", "initial")


class FoxGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        init_repo(self.repo)
        fox.ROOT = self.repo

    def test_protected_paths_cover_credentials_and_runtime(self):
        blocked = [
            ".env",
            ".env.local",
            "cloudflare.env",
            ".config/fox-ai/config.json",
            ".fox-ai/backups/a.txt",
            "credentials.json",
            "keys/server.pem",
            "cache/session.sqlite3",
            "__pycache__/module.pyc",
        ]
        for path in blocked:
            with self.subTest(path=path):
                self.assertIsNotNone(fox.protected_path_reason(path))
        self.assertIsNone(fox.protected_path_reason("src/tokenizer.py"))
        self.assertIsNone(fox.protected_path_reason("src/config.py"))

    def test_repository_branch_remote_and_status_are_detected(self):
        bare = self.base / "remote.git"
        run_git(self.base, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "origin", str(bare))
        run_git(self.repo, "push", "-u", "origin", "main")

        info = fox._git_info()
        self.assertEqual(info["root"], self.repo.resolve())
        self.assertEqual(info["branch"], "main")
        self.assertEqual(info["remote"], "origin")
        self.assertEqual(info["remote_branch"], "main")

    def test_safe_staging_excludes_env_and_runtime_files(self):
        (self.repo / "app.py").write_text("print('two')\n", encoding="utf-8")
        (self.repo / ".env").write_text("DO_NOT_STAGE=yes\n", encoding="utf-8")
        # Simulate a sensitive file that was already force-staged before Fox AI.
        run_git(self.repo, "add", "-f", ".env")
        runtime_dir = self.repo / ".fox-ai"
        runtime_dir.mkdir()
        (runtime_dir / "state.json").write_text("{}\n", encoding="utf-8")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(fox._stage_safe_changes(self.repo))

        staged = run_git(self.repo, "diff", "--cached", "--name-only").splitlines()
        self.assertEqual(staged, ["app.py"])
        self.assertTrue((self.repo / ".env").exists())
        self.assertTrue((runtime_dir / "state.json").exists())

    def test_secret_scanner_unstages_suspicious_literal(self):
        # Build the text in pieces so this test file itself contains no usable key.
        suspicious = '"api_' + 'key": "' + ('live-value-' * 3) + '"\n'
        target = self.repo / "settings.txt"
        target.write_text(suspicious, encoding="utf-8")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(fox._stage_safe_changes(self.repo))

        self.assertEqual(run_git(self.repo, "diff", "--cached", "--name-only"), "")
        self.assertTrue(target.exists())

    def test_diff_never_prints_sensitive_file_contents(self):
        value = "value-that-must-not-appear"
        (self.repo / ".env").write_text(f"PASSWORD={value}\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(fox.git_diff())
        self.assertNotIn(value, output.getvalue())

    def test_push_command_does_not_commit_without_explicit_confirmation(self):
        bare = self.base / "remote.git"
        run_git(self.base, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "origin", str(bare))
        run_git(self.repo, "push", "-u", "origin", "main")
        before = run_git(self.repo, "rev-parse", "HEAD")
        (self.repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

        with mock.patch("builtins.input", return_value="cancel"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(fox.git_push())

        self.assertEqual(run_git(self.repo, "rev-parse", "HEAD"), before)
        self.assertEqual(run_git(self.repo, "diff", "--cached", "--name-only"), "")

    def test_explicit_stage_commit_and_push_flow(self):
        bare = self.base / "remote.git"
        run_git(self.base, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "origin", str(bare))
        run_git(self.repo, "push", "-u", "origin", "main")
        (self.repo / "app.py").write_text("print('published')\n", encoding="utf-8")

        answers = ["stage", "safe update", "commit", "push"]
        with mock.patch("builtins.input", side_effect=answers):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(fox.git_push())

        local_head = run_git(self.repo, "rev-parse", "HEAD")
        remote_head = run_git(self.repo, "rev-parse", "origin/main")
        self.assertEqual(local_head, remote_head)
        self.assertEqual(run_git(self.repo, "log", "-1", "--pretty=%s"), "safe update")

    def test_outgoing_audit_blocks_sensitive_files_in_earlier_commits(self):
        bare = self.base / "remote.git"
        run_git(self.base, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "origin", str(bare))
        run_git(self.repo, "push", "-u", "origin", "main")

        protected = self.repo / ".env"
        protected.write_text("SAFE_TEST_VALUE=not-a-real-secret\n", encoding="utf-8")
        run_git(self.repo, "add", "-f", ".env")
        run_git(self.repo, "commit", "-m", "bad local commit")

        info = fox._git_info()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(fox._refresh_remote(info))
        commits, error = fox._outgoing_commits(self.repo, "origin")
        self.assertEqual(error, "")
        findings, error = fox._audit_outgoing(self.repo, commits)
        self.assertEqual(error, "")
        self.assertTrue(any(path == ".env" for _, path, _ in findings))

    def test_sync_fetches_and_fast_forwards_after_confirmation(self):
        bare = self.base / "remote.git"
        run_git(self.base, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "origin", str(bare))
        run_git(self.repo, "push", "-u", "origin", "main")

        other = self.base / "other"
        run_git(self.base, "clone", "--branch", "main", str(bare), str(other))
        run_git(other, "config", "user.name", "Other Test")
        run_git(other, "config", "user.email", "other@example.invalid")
        (other / "app.py").write_text("print('remote')\n", encoding="utf-8")
        run_git(other, "add", "app.py")
        run_git(other, "commit", "-m", "remote update")
        run_git(other, "push", "origin", "main")

        with mock.patch("builtins.input", return_value="sync"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(fox.git_sync())

        self.assertEqual(
            (self.repo / "app.py").read_text(encoding="utf-8"),
            "print('remote')\n",
        )
        self.assertEqual(
            run_git(self.repo, "rev-parse", "HEAD"),
            run_git(self.repo, "rev-parse", "origin/main"),
        )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FoxProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        fox.ROOT = PROJECT
        self.account_id = "a" * 32
        self.cf_token = "cf-test-" + ("s" * 24)
        self.gemini_key = "gemini-test-" + ("k" * 24)
        self.gemini_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": self.gemini_key,
            "model": "gemini-test-model",
        }

    def test_cloudflare_credentials_are_read_from_secure_local_file(self):
        credential_file = self.base / "cloudflare.env"
        credential_file.write_text(
            "# local only\n"
            f"CLOUDFLARE_ACCOUNT_ID={self.account_id}\n"
            f"CLOUDFLARE_API_TOKEN='{self.cf_token}'\n",
            encoding="utf-8",
        )
        credential_file.chmod(0o600)

        loaded = fox.load_cloudflare_credentials(
            path=credential_file,
            environ={},
        )
        self.assertEqual(loaded["CLOUDFLARE_ACCOUNT_ID"], self.account_id)
        self.assertEqual(loaded["CLOUDFLARE_API_TOKEN"], self.cf_token)

    def test_cloudflare_credentials_never_enter_context_or_backup(self):
        fox.ROOT = self.base
        credential_file = self.base / "cloudflare.env"
        credential_file.write_text(
            f"CLOUDFLARE_ACCOUNT_ID={self.account_id}\n"
            f"CLOUDFLARE_API_TOKEN={self.cf_token}\n",
            encoding="utf-8",
        )
        credential_file.chmod(0o600)

        self.assertNotIn(self.account_id, fox.project_context())
        self.assertNotIn(self.cf_token, fox.project_context())
        action = {
            "actions": [{
                "type": "edit",
                "path": "cloudflare.env",
                "old": self.cf_token,
                "new": "replacement",
            }]
        }
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(fox.apply_actions(action))
        self.assertIn(
            self.cf_token,
            credential_file.read_text(encoding="utf-8"),
        )
        self.assertFalse((self.base / ".fox-ai").exists())

    def test_environment_credentials_work_without_a_file(self):
        loaded = fox.load_cloudflare_credentials(
            path=self.base / "missing.env",
            environ={
                "CLOUDFLARE_ACCOUNT_ID": self.account_id,
                "CLOUDFLARE_API_TOKEN": self.cf_token,
            },
        )
        self.assertEqual(loaded["CLOUDFLARE_ACCOUNT_ID"], self.account_id)
        self.assertEqual(loaded["CLOUDFLARE_API_TOKEN"], self.cf_token)

    def test_cloudflare_credentials_are_removed_from_json_config(self):
        config_dir = self.base / "config"
        config_file = config_dir / "config.json"
        with mock.patch.object(fox, "CONFIG_DIR", config_dir), mock.patch.object(
            fox, "CONFIG_FILE", config_file
        ):
            fox.save_config({
                "base_url": self.gemini_config["base_url"],
                "api_key": self.gemini_key,
                "model": self.gemini_config["model"],
                "CLOUDFLARE_ACCOUNT_ID": self.account_id,
                "CLOUDFLARE_API_TOKEN": self.cf_token,
            })
            saved = config_file.read_text(encoding="utf-8")
            loaded = fox.load_config()

        self.assertNotIn(self.account_id, saved)
        self.assertNotIn(self.cf_token, saved)
        self.assertNotIn("CLOUDFLARE_ACCOUNT_ID", loaded)
        self.assertNotIn("CLOUDFLARE_API_TOKEN", loaded)
        self.assertEqual(loaded["api_key"], self.gemini_key)

    def test_insecure_cloudflare_file_permissions_are_rejected(self):
        credential_file = self.base / "cloudflare.env"
        credential_file.write_text(
            f"CLOUDFLARE_ACCOUNT_ID={self.account_id}\n"
            f"CLOUDFLARE_API_TOKEN={self.cf_token}\n",
            encoding="utf-8",
        )
        credential_file.chmod(0o644)
        with self.assertRaises(fox.ProviderConfigurationError):
            fox.load_cloudflare_credentials(path=credential_file, environ={})

    def test_cloudflare_is_the_primary_provider(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return FakeResponse({
                "choices": [{"message": {"content": "cloudflare-ok"}}]
            })

        credentials = {
            "CLOUDFLARE_ACCOUNT_ID": self.account_id,
            "CLOUDFLARE_API_TOKEN": self.cf_token,
        }
        with mock.patch.object(
            fox, "load_cloudflare_credentials", return_value=credentials
        ), mock.patch.object(
            fox, "load_config", return_value=self.gemini_config
        ), mock.patch.object(
            fox.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            content, provider = fox.request_chat_completion([
                {"role": "user", "content": "hello"}
            ])

        self.assertEqual(content, "cloudflare-ok")
        self.assertEqual(provider, "Cloudflare Workers AI")
        self.assertEqual(len(requests), 1)
        self.assertIn("/ai/v1/chat/completions", requests[0].full_url)
        self.assertEqual(
            json.loads(requests[0].data.decode("utf-8"))["model"],
            fox.CLOUDFLARE_MODEL,
        )

    def test_cloudflare_failure_falls_back_to_gemini_without_secret_leak(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            if len(requests) == 1:
                raise fox.urllib.error.HTTPError(
                    request.full_url, 429, "busy", None, io.BytesIO(b"busy")
                )
            return FakeResponse({
                "choices": [{"message": {"content": "fallback-ok"}}]
            })

        credentials = {
            "CLOUDFLARE_ACCOUNT_ID": self.account_id,
            "CLOUDFLARE_API_TOKEN": self.cf_token,
        }
        output = io.StringIO()
        with mock.patch.object(
            fox, "load_cloudflare_credentials", return_value=credentials
        ), mock.patch.object(
            fox, "load_config", return_value=self.gemini_config
        ), mock.patch.object(
            fox.urllib.request, "urlopen", side_effect=fake_urlopen
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            content, provider = fox.request_chat_completion([
                {"role": "user", "content": "hello"}
            ])

        self.assertEqual(content, "fallback-ok")
        self.assertEqual(provider, "Gemini")
        self.assertEqual(len(requests), 2)
        first_payload = requests[0].data.decode("utf-8")
        self.assertIn(fox.CLOUDFLARE_MODEL, first_payload)
        self.assertNotIn(self.account_id, first_payload)
        self.assertNotIn(self.cf_token, first_payload)
        combined_output = output.getvalue()
        self.assertNotIn(self.account_id, combined_output)
        self.assertNotIn(self.cf_token, combined_output)
        self.assertNotIn(self.gemini_key, combined_output)

    def test_secret_scanner_detects_cloudflare_values(self):
        token_text = "CLOUDFLARE_API_" + "TOKEN=" + self.cf_token
        account_text = "CLOUDFLARE_ACCOUNT_" + "ID=" + self.account_id
        self.assertIn("literal API_TOKEN", fox._secret_labels(token_text))
        self.assertIn("literal ACCOUNT_ID", fox._secret_labels(account_text))

    def test_status_shows_active_provider_without_credentials(self):
        credentials = {
            "CLOUDFLARE_ACCOUNT_ID": self.account_id,
            "CLOUDFLARE_API_TOKEN": self.cf_token,
        }
        output = io.StringIO()
        with mock.patch.object(
            fox, "load_cloudflare_credentials", return_value=credentials
        ), mock.patch.object(
            fox, "load_config", return_value=self.gemini_config
        ), mock.patch.object(
            fox, "collect_files", return_value=[]
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            fox.status()

        displayed = output.getvalue()
        self.assertIn("Active Provider: Cloudflare Workers AI", displayed)
        self.assertIn(fox.CLOUDFLARE_MODEL, displayed)
        self.assertIn("Gemini fallback: configured", displayed)
        self.assertNotIn(self.account_id, displayed)
        self.assertNotIn(self.cf_token, displayed)
        self.assertNotIn(self.gemini_key, displayed)

    def test_clear_error_when_no_provider_is_configured(self):
        with mock.patch.object(
            fox, "load_cloudflare_credentials", return_value={}
        ), mock.patch.object(
            fox,
            "load_config",
            return_value={"base_url": "", "api_key": "", "model": ""},
        ):
            with self.assertRaisesRegex(
                fox.NoProviderAvailableError, "هیچ Provider قابل استفاده نیست"
            ):
                fox.request_chat_completion([
                    {"role": "user", "content": "hello"}
                ])


if __name__ == "__main__":
    unittest.main()
