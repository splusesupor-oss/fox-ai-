import contextlib
import importlib.util
import io
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


if __name__ == "__main__":
    unittest.main()
