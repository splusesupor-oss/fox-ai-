#!/usr/bin/env python3

import os
import sys
import json
import shutil
import difflib
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

HOME = Path.home()
CONFIG_DIR = HOME / ".config" / "fox-ai"
CONFIG_FILE = CONFIG_DIR / "config.json"
BACKUP_DIR = Path(".fox-ai") / "backups"

DEFAULT_IGNORE = {
    ".git", ".fox-ai", "__pycache__", ".venv", "venv",
    "node_modules", ".idea", ".gradle"
}

MAX_FILE = 120_000


def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass

    return {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini"
    }


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2)
    )
    os.chmod(CONFIG_FILE, 0o600)


def project_root():
    p = Path.cwd().resolve()

    # اگر داخل git هستیم، ریشه پروژه را پیدا کن
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=p,
            capture_output=True,
            text=True
        )
        if r.returncode == 0:
            return Path(r.stdout.strip()).resolve()
    except Exception:
        pass

    return p


ROOT = project_root()


def safe_path(rel):
    target = (ROOT / rel).resolve()

    try:
        target.relative_to(ROOT)
    except ValueError:
        raise Exception(f"مسیر خارج از پروژه ممنوع است: {rel}")

    return target


def should_ignore(path):
    return any(part in DEFAULT_IGNORE for part in path.parts)


def collect_files():
    result = []

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(ROOT)

        if should_ignore(rel):
            continue

        try:
            if p.stat().st_size > MAX_FILE:
                continue

            result.append(str(rel))
        except Exception:
            pass

    return sorted(result)


def read_file(path):
    p = safe_path(path)

    if not p.exists():
        raise Exception(f"فایل وجود ندارد: {path}")

    if p.stat().st_size > MAX_FILE:
        raise Exception(f"فایل خیلی بزرگ است: {path}")

    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")


def project_context():
    files = collect_files()

    chunks = []

    for name in files:
        try:
            content = read_file(name)
            chunks.append(
                f"\n===== FILE: {name} =====\n{content}"
            )
        except Exception:
            pass

    text = "\n".join(chunks)

    # جلوگیری از ارسال context بیش از حد بزرگ
    if len(text) > 300_000:
        text = text[:300_000] + "\n[CONTEXT TRUNCATED]"

    return text


SYSTEM = r"""
تو Fox AI Coding Agent هستی.

وظیفه:
- پروژه را بررسی کن.
- درخواست کاربر را بفهم.
- فقط فایل‌هایی را تغییر بده که واقعاً لازم هستند.
- اگر لازم است فایل جدید بساز.
- کد موجود را تا حد ممکن حفظ کن.
- تغییرات را کوچک و دقیق انجام بده.
- هیچ shell command تولید نکن.
- هیچ مسیر خارج از پروژه درخواست نکن.

خروجی تو باید فقط JSON معتبر باشد.

ساختار:

{
  "summary": "توضیح کوتاه",
  "actions": [
    {
      "type": "edit",
      "path": "path/file.py",
      "old": "متن دقیق موجود",
      "new": "متن جایگزین"
    },
    {
      "type": "create",
      "path": "new_file.py",
      "content": "محتوای کامل فایل"
    },
    {
      "type": "delete",
      "path": "file.py"
    }
  ],
  "tests": [
    "python -m pytest tests/test_x.py"
  ]
}

قوانین:
- برای edit مقدار old باید دقیقاً در فایل موجود باشد.
- برای فایل جدید از create استفاده کن.
- برای حذف از delete استفاده کن.
- اگر تغییر لازم نیست actions را خالی کن.
- JSON را داخل markdown قرار نده.
"""


def ask_ai(user_request):
    cfg = load_config()

    api_key = cfg.get("api_key", "").strip()
    base_url = cfg.get("base_url", "").rstrip("/")
    model = cfg.get("model", "").strip()

    if not api_key:
        print("""
❌ API Key تنظیم نشده.

این دستور را بزن:

fox-ai config

بعد API Key را وارد کن.
""")
        return None

    context = project_context()

    prompt = f"""
PROJECT ROOT:
{ROOT}

PROJECT FILES:
{context}

USER REQUEST:
{user_request}

حالا بهترین تغییرات لازم را به صورت JSON برگردان.
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    url = base_url + "/chat/completions"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode())

        content = data["choices"][0]["message"]["content"].strip()

        # استخراج JSON حتی اگر مدل قبل/بعدش توضیح داده باشد
        import re

        content = content.strip()

        # markdown code fence
        if "```" in content:
            blocks = re.findall(r"```(?:json)?\\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
            if blocks:
                content = blocks[0].strip()

        # پیدا کردن اولین object JSON
        if not content.startswith("{"):
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end > start:
                content = content[start:end + 1]

        return json.loads(content)

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"\n❌ API Error {e.code}\n{body}")
        return None

    except Exception as e:
        print(f"\n❌ خطا در ارتباط با AI:\n{e}")
        return None


def backup_file(path, timestamp):
    if not path.exists():
        return

    dest = ROOT / BACKUP_DIR / timestamp / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def show_diff(old, new, filename):
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=filename,
        tofile=filename
    )

    print("\n".join(diff))


def apply_actions(result):
    actions = result.get("actions", [])

    if not actions:
        print("\n✅ تغییری لازم نیست.")
        return True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("پیشنهاد Fox AI")
    print("=" * 60)

    prepared = []

    for action in actions:
        typ = action.get("type")
        rel = action.get("path")

        try:
            path = safe_path(rel)
        except Exception as e:
            print(f"❌ {e}")
            return False

        if typ == "edit":
            if not path.exists():
                print(f"❌ فایل پیدا نشد: {rel}")
                return False

            old = path.read_text(encoding="utf-8", errors="replace")
            expected = action.get("old", "")
            new = action.get("new", "")

            if expected not in old:
                print(f"❌ متن old در فایل پیدا نشد: {rel}")
                return False

            if old.count(expected) != 1:
                print(f"❌ old باید دقیقاً یک بار در فایل باشد: {rel}")
                return False

            updated = old.replace(expected, new, 1)

            print(f"\n✏️ EDIT: {rel}")
            show_diff(old, updated, rel)

            prepared.append(("edit", path, updated))

        elif typ == "create":
            content = action.get("content", "")

            if path.exists():
                print(f"❌ فایل از قبل وجود دارد: {rel}")
                return False

            print(f"\n➕ CREATE: {rel}")
            print("-" * 60)
            print(content[:5000])
            if len(content) > 5000:
                print("...")

            prepared.append(("create", path, content))

        elif typ == "delete":
            if not path.exists():
                print(f"⚠️ فایل وجود ندارد: {rel}")
                continue

            print(f"\n🗑️ DELETE: {rel}")
            prepared.append(("delete", path, None))

        else:
            print(f"❌ action ناشناخته: {typ}")
            return False

    print("\n" + "=" * 60)

    answer = input("اعمال تغییرات؟ [y/N]: ").strip().lower()

    if answer not in ("y", "yes", "بله"):
        print("❎ لغو شد.")
        return False

    # backup
    for typ, path, content in prepared:
        if path.exists():
            backup_file(path, timestamp)

    # apply
    for typ, path, content in prepared:
        if typ == "edit":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        elif typ == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        elif typ == "delete":
            path.unlink()

    print(f"\n✅ تغییرات اعمال شد.")
    print(f"💾 Backup: .fox-ai/backups/{timestamp}")

    tests = result.get("tests", [])

    if tests:
        print("\n🧪 تست‌های پیشنهادی:")
        for t in tests:
            print("  ", t)

        run = input("\nاولین تست اجرا شود؟ [y/N]: ").strip().lower()

        if run in ("y", "yes", "بله"):
            try:
                cmd = tests[0]
                print("\n▶", cmd)

                p = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=ROOT,
                    text=True
                )

                if p.returncode == 0:
                    print("\n✅ تست موفق بود.")
                else:
                    print(f"\n❌ تست شکست خورد. code={p.returncode}")

            except Exception as e:
                print("❌ اجرای تست:", e)

    return True


def undo():
    base = ROOT / BACKUP_DIR

    if not base.exists():
        print("❌ هیچ backupای وجود ندارد.")
        return

    backups = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        reverse=True
    )

    if not backups:
        print("❌ هیچ backupای وجود ندارد.")
        return

    latest = backups[0]

    print(f"آخرین backup: {latest.name}")

    answer = input("برگردانم؟ [y/N]: ").strip().lower()

    if answer not in ("y", "yes", "بله"):
        return

    for src in latest.rglob("*"):
        if src.is_file():
            rel = src.relative_to(latest)
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    print("✅ آخرین backup برگردانده شد.")


def configure():
    cfg = load_config()

    print("\nFox AI Configuration\n")

    base = input(
        f"Base URL [{cfg.get('base_url')}]: "
    ).strip()

    if base:
        cfg["base_url"] = base.rstrip("/")

    key = input("API Key: ").strip()

    if key:
        cfg["api_key"] = key

    model = input(
        f"Model [{cfg.get('model')}]: "
    ).strip()

    if model:
        cfg["model"] = model

    save_config(cfg)

    print("\n✅ تنظیمات ذخیره شد.")
    print(CONFIG_FILE)


def status():
    print("\n🦊 Fox AI")
    print("Project:", ROOT)

    files = collect_files()

    print("Files:", len(files))

    cfg = load_config()

    print("API:", "configured" if cfg.get("api_key") else "NOT configured")
    print("Model:", cfg.get("model"))
    print("Base:", cfg.get("base_url"))


def run_request(req):
    print("\n🦊 Fox AI")
    print("📂 Project:", ROOT)
    print("\n🧠 در حال بررسی پروژه...")

    result = ask_ai(req)

    if not result:
        return

    print("\n🤖", result.get("summary", "No summary"))

    apply_actions(result)



def chat():
    print("""
🦊 Fox AI Chat
پروژه: {}
دستورهای داخلی:
  /exit   خروج
  /status وضعیت پروژه
  /undo   برگشت آخرین backup

هر درخواست دیگری را مستقیم به AI می‌فرستیم.
""".format(ROOT))

    history = []

    while True:
        try:
            user = input("\n🦊 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 خروج.")
            break

        if not user:
            continue

        cmd = user.lower()

        if cmd in ("/exit", "exit", "quit", "خروج"):
            print("👋 خروج از Fox AI Chat")
            break

        if cmd == "/status":
            status()
            continue

        if cmd == "/undo":
            undo()
            continue

        history.append({"role": "user", "content": user})

        context = project_context()

        cfg = load_config()
        api_key = cfg.get("api_key", "").strip()
        base_url = cfg.get("base_url", "").rstrip("/")
        model = cfg.get("model", "").strip()

        if not api_key:
            print("❌ API Key تنظیم نشده. بزن: fox-ai config")
            continue

        chat_system = SYSTEM + r"""

این یک گفت‌وگوی چندمرحله‌ای است.
اگر کاربر درباره پروژه سؤال کرد، فایل‌های پروژه را بررسی کن.
اگر درخواست تغییر داد، actionهای دقیق تولید کن.
اگر کاربر گفت تست کن، تست مناسب پیشنهاد بده.
اگر درخواست فقط توضیح بود، actions را خالی بگذار.
"""

        messages = [{"role": "system", "content": chat_system}]

        # حفظ چند پیام اخیر برای جلوگیری از بزرگ شدن بیش از حد context
        for item in history[-10:]:
            messages.append(item)

        messages.append({
            "role": "user",
            "content": f"""
PROJECT ROOT:
{ROOT}

CURRENT PROJECT FILES:
{context}

CURRENT REQUEST:
{user}
"""
        })

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1
        }

        url = base_url + "/chat/completions"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key
            },
            method="POST"
        )

        try:
            print("\n🧠 در حال فکر کردن...")

            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.loads(response.read().decode())

            content = data["choices"][0]["message"]["content"].strip()

            print("\n🤖", content)

            # اگر پاسخ شامل actionهای JSON باشد
            try:
                clean = content

                if "```" in clean:
                    blocks = re.findall(
                        r"```(?:json)?\s*(.*?)```",
                        clean,
                        re.DOTALL | re.IGNORECASE
                    )
                    if blocks:
                        clean = blocks[0].strip()

                if not clean.startswith("{"):
                    start = clean.find("{")
                    end = clean.rfind("}")
                    if start != -1 and end > start:
                        clean = clean[start:end + 1]

                result = json.loads(clean)

                if isinstance(result, dict) and "actions" in result:
                    print("\n" + "=" * 60)
                    print("تغییرات پیشنهادی")
                    print("=" * 60)

                    apply_actions(result)

            except Exception:
                # پاسخ معمولی بوده و action قابل اجرا ندارد
                pass

        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            print(f"\n❌ API Error {e.code}\n{body}")

        except Exception as e:
            print(f"\n❌ خطا در ارتباط با AI:\n{e}")


def main():
    global ROOT
    ROOT = project_root()

    args = sys.argv[1:]

    if not args:
        print("""
🦊 Fox AI Coding Agent

استفاده:

  fox-ai "درخواست شما"

  fox-ai config
  fox-ai status
  fox-ai undo

مثال:

  fox-ai "فیلتر اسم حسین را بررسی کن و اگر مشکل دارد اصلاحش کن و تست اضافه کن"
""")
        return

    command = args[0].lower()

    if command == "config":
        configure()
        return

    if command == "status":
        status()
        return

    if command == "undo":
        undo()
        return

    if command == "chat":
        chat()
        return

    run_request(" ".join(args))



if __name__ == "__main__":
    if len(sys.argv) >= 2:
        command = sys.argv[1]

        if command == "git":
            git_status()
            raise SystemExit

        if command == "diff":
            git_diff()
            raise SystemExit

        if command == "push":
            git_push()
            raise SystemExit

        if command == "sync":
            git_sync()
            raise SystemExit

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        command = sys.argv[1]

        if command == "git":
            git_status()
            raise SystemExit

        if command == "diff":
            git_diff()
            raise SystemExit

        if command == "push":
            git_push()
            raise SystemExit

        if command == "sync":
            git_sync()
            raise SystemExit

if __name__ == "__main__":
    main()

# --- GitHub integration ---
def git_cmd(*args):
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=project_root(),
            text=True,
            capture_output=True,
            timeout=30,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

def git_status():
    code, branch, err = git_cmd("branch", "--show-current")
    if code != 0:
        print(f"❌ Git: {err}")
        return

    _, remote, _ = git_cmd("remote", "get-url", "origin")
    _, status, _ = git_cmd("status", "--short")

    print("🦊 GitHub")
    print(f"Project: {project_root()}")
    print(f"Branch: {branch or '(unknown)'}")
    print(f"Remote: {remote or '(not configured)'}")
    print("Status:")
    print(status or "  clean")

def git_diff():
    _, diff, err = git_cmd("diff", "--")
    if err:
        print(f"❌ {err}")
        return
    print(diff or "✅ No local changes.")

def git_push():
    root = project_root()

    code, branch, err = git_cmd("branch", "--show-current")
    if code != 0 or not branch:
        print("❌ شاخه Git پیدا نشد.")
        return

    code, remote, err = git_cmd("remote", "get-url", "origin")
    if code != 0 or not remote:
        print("❌ remote به GitHub تنظیم نشده.")
        return

    print(f"🦊 Branch: {branch}")
    print(f"🌐 Remote: {remote}")

    _, status, _ = git_cmd("status", "--short")
    if not status:
        print("✅ تغییری برای ارسال وجود ندارد.")
        return

    print("\n📋 تغییرات:")
    git_diff()

    answer = input("\nCommit و Push انجام شود؟ [y/N]: ").strip().lower()
    if answer != "y":
        print("❎ لغو شد.")
        return

    # Protect sensitive files from accidental staging.
    protected = [
        ".env",
        ".env.*",
        "*.key",
        "*.pem",
        "*.token",
        "*secret*",
        "*credential*",
    ]

    for pattern in protected:
        git_cmd("reset", "--", pattern)

    code, out, err = git_cmd("add", "-A")
    if code != 0:
        print(f"❌ git add: {err}")
        return

    # Remove sensitive files from the index if they were accidentally staged.
    for pattern in protected:
        git_cmd("reset", "--", pattern)

    message = input("📝 پیام Commit: ").strip()
    if not message:
        message = "Update project via Fox AI"

    code, out, err = git_cmd("commit", "-m", message)
    if code != 0:
        print(f"❌ git commit: {err or out}")
        return

    print("✅ Commit ساخته شد.")

    code, out, err = git_cmd("push", "origin", branch)
    if code != 0:
        print(f"❌ git push: {err or out}")
        return

    print("🚀 با موفقیت به GitHub Push شد.")

def git_sync():
    print("🦊 Git Sync")
    code, branch, err = git_cmd("branch", "--show-current")
    if code != 0 or not branch:
        print("❌ شاخه Git پیدا نشد.")
        return

    _, status, _ = git_cmd("status", "--short")
    if status:
        print("⚠️ تغییرات محلی داری؛ قبل از pull آنها را بررسی کن:")
        print(status)
        return

    code, out, err = git_cmd("pull", "--ff-only", "origin", branch)
    if code != 0:
        print(f"❌ git pull: {err or out}")
        return

    print(out or "✅ پروژه با GitHub همگام شد.")

