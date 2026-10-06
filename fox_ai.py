#!/usr/bin/env python3

import os
import sys
import json
import shutil
import difflib
import getpass
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path, PurePosixPath
from datetime import datetime

HOME = Path.home()
CONFIG_DIR = HOME / ".config" / "fox-ai"
CONFIG_FILE = CONFIG_DIR / "config.json"
CLOUDFLARE_ENV_FILE = CONFIG_DIR / "cloudflare.env"
BACKUP_DIR = Path(".fox-ai") / "backups"

# Verified against Cloudflare's official model catalog on 2026-10-06:
# https://developers.cloudflare.com/workers-ai/models/glm-4.7-flash/
CLOUDFLARE_MODEL = "@cf/zai-org/glm-4.7-flash"
# Official OpenAI-compatible endpoint documentation:
# https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/
CLOUDFLARE_API_ROOT = "https://api.cloudflare.com/client/v4/accounts"
CLOUDFLARE_CREDENTIAL_NAMES = (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
)

DEFAULT_IGNORE = {
    ".git", ".fox-ai", "__pycache__", ".venv", "venv",
    "node_modules", ".idea", ".gradle", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox",
}

# These files must never be sent to the AI, staged, committed, or pushed by
# Fox AI.  This is intentionally enforced in Python as well as .gitignore,
# because .gitignore does not protect files that were already tracked/staged.
PROTECTED_DIR_NAMES = {
    ".fox-ai", ".git", ".config", ".cache", ".ssh", ".gnupg", "secrets",
    "credentials", "runtime", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".venv", "venv",
    "node_modules", "coverage", "htmlcov", "build", "dist", "tmp",
    "temp", "logs",
}
PROTECTED_FILE_NAMES = {
    ".env", "cloudflare.env", ".netrc", ".npmrc", ".pypirc", "credentials",
    "credentials.json", "secrets.json", "service-account.json",
    "service_account.json", "runtime.json", "runtime-data.json",
    "state.json", "session.json", ".session", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "known_hosts", ".coverage",
}
PROTECTED_SUFFIXES = {
    ".key", ".pem", ".p12", ".pfx", ".jks", ".keystore", ".token",
    ".log", ".pid", ".sock", ".sqlite", ".sqlite3", ".db", ".pyc",
    ".pyo", ".tmp", ".temp", ".apk", ".aab",
}

MAX_FILE = 120_000
MAX_SECRET_SCAN = 2_000_000


def _without_cloudflare_credentials(cfg):
    """Keep Cloudflare credentials out of the legacy JSON config."""
    if not isinstance(cfg, dict):
        return {}
    blocked = {name.casefold() for name in CLOUDFLARE_CREDENTIAL_NAMES}
    return {
        key: value
        for key, value in cfg.items()
        if str(key).casefold() not in blocked
        and not str(key).casefold().startswith("cloudflare_")
    }


def load_config():
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return _without_cloudflare_credentials(data)
        except Exception:
            pass

    return {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini"
    }


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    safe_config = _without_cloudflare_credentials(cfg)
    CONFIG_FILE.write_text(
        json.dumps(safe_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(CONFIG_FILE, 0o600)


class ProviderConfigurationError(Exception):
    """A provider is not safely/configurably available."""


class ProviderRequestError(Exception):
    """A provider request failed without exposing response bodies or secrets."""

    def __init__(self, provider, reason, fallbackable=False):
        super().__init__(reason)
        self.provider = provider
        self.reason = reason
        self.fallbackable = fallbackable


class NoProviderAvailableError(Exception):
    """No configured provider could complete a request."""


def _parse_cloudflare_env(text):
    """Parse only the two supported keys; never execute or expand env content."""
    values = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ProviderConfigurationError(
                f"ساختار cloudflare.env در خط {line_number} نامعتبر است."
            )

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in CLOUDFLARE_CREDENTIAL_NAMES:
            # Ignore unrelated local settings instead of exposing their values.
            continue

        raw_value = raw_value.strip()
        if raw_value.startswith(("\"", "'")):
            quote = raw_value[0]
            if len(raw_value) < 2 or raw_value[-1] != quote:
                raise ProviderConfigurationError(
                    f"مقدار نقل‌قول‌شده در خط {line_number} کامل نیست."
                )
            value = raw_value[1:-1]
        else:
            # Allow an inline comment only when it starts after whitespace.
            value = re.split(r"\s+#", raw_value, maxsplit=1)[0].strip()

        if "\x00" in value or "\n" in value or "\r" in value:
            raise ProviderConfigurationError(
                f"مقدار cloudflare.env در خط {line_number} نامعتبر است."
            )
        values[key] = value

    return values


def load_cloudflare_credentials(path=None, environ=None):
    """Load Cloudflare credentials without copying them into app config/logs."""
    env = os.environ if environ is None else environ
    env_values = {
        name: str(env.get(name, "")).strip()
        for name in CLOUDFLARE_CREDENTIAL_NAMES
    }

    # Complete environment credentials take precedence and do not require
    # touching a local credential file.
    if all(env_values.values()):
        values = env_values
    else:
        credential_path = Path(path or CLOUDFLARE_ENV_FILE).expanduser()
        values = {}
        if credential_path.exists():
            try:
                file_stat = credential_path.stat()
                if not credential_path.is_file():
                    raise ProviderConfigurationError(
                        "مسیر Cloudflare credential یک فایل عادی نیست."
                    )
                if file_stat.st_size > 64_000:
                    raise ProviderConfigurationError(
                        "فایل Cloudflare credential بیش از حد بزرگ است."
                    )
                if file_stat.st_mode & 0o077:
                    raise ProviderConfigurationError(
                        "دسترسی cloudflare.env ناامن است؛ chmod 600 اجرا کنید."
                    )
                file_values = _parse_cloudflare_env(
                    credential_path.read_text(encoding="utf-8")
                )
                values.update(file_values)
            except ProviderConfigurationError:
                raise
            except (OSError, UnicodeError) as exc:
                raise ProviderConfigurationError(
                    "خواندن امن cloudflare.env ممکن نیست."
                ) from exc

        # Any explicitly supplied environment value overrides the file value.
        for name, value in env_values.items():
            if value:
                values[name] = value

    account_id = values.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = values.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id and not api_token:
        return {}
    if not account_id or not api_token:
        raise ProviderConfigurationError(
            "هر دو متغیر Cloudflare باید تنظیم شوند."
        )
    if not re.fullmatch(r"[0-9a-fA-F]{32}", account_id):
        raise ProviderConfigurationError(
            "CLOUDFLARE_ACCOUNT_ID باید یک شناسه معتبر ۳۲ کاراکتری باشد."
        )

    return {
        "CLOUDFLARE_ACCOUNT_ID": account_id,
        "CLOUDFLARE_API_TOKEN": api_token,
    }


def _cloudflare_provider():
    credentials = load_cloudflare_credentials()
    if not credentials:
        return None
    account_id = credentials["CLOUDFLARE_ACCOUNT_ID"]
    return {
        "id": "cloudflare",
        "name": "Cloudflare Workers AI",
        "model": CLOUDFLARE_MODEL,
        "url": (
            f"{CLOUDFLARE_API_ROOT}/{account_id}/ai/v1/chat/completions"
        ),
        "api_key": credentials["CLOUDFLARE_API_TOKEN"],
    }


def _gemini_provider():
    """Build the fallback from the existing, backward-compatible config."""
    cfg = load_config()
    api_key = str(cfg.get("api_key", "")).strip()
    if not api_key:
        return None

    base_url = str(cfg.get("base_url", "")).rstrip("/")
    model = str(cfg.get("model", "")).strip()
    if not base_url or not model:
        raise ProviderConfigurationError(
            "تنظیمات Gemini ناقص است؛ fox-ai config را اجرا کنید."
        )
    return {
        "id": "gemini",
        "name": "Gemini",
        "model": model,
        "url": base_url + "/chat/completions",
        "api_key": api_key,
    }


def provider_chain():
    """Return Cloudflare first and the existing Gemini config second."""
    providers = []
    notices = []

    try:
        cloudflare = _cloudflare_provider()
        if cloudflare:
            providers.append(cloudflare)
        else:
            notices.append("Cloudflare تنظیم نشده است")
    except ProviderConfigurationError as exc:
        notices.append(f"Cloudflare: {exc}")

    try:
        gemini = _gemini_provider()
        if gemini:
            providers.append(gemini)
        else:
            notices.append("Gemini تنظیم نشده است")
    except ProviderConfigurationError as exc:
        notices.append(f"Gemini: {exc}")

    return providers, notices


def _request_provider(provider, messages, temperature=0.1):
    payload = {
        "model": provider["model"],
        "messages": messages,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        provider["url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + provider["api_key"],
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # An HTTP failure is isolated to the primary provider; the same
        # OpenAI-compatible request may still succeed with Gemini.
        fallbackable = provider["id"] == "cloudflare"
        raise ProviderRequestError(
            provider["name"], f"HTTP {exc.code}", fallbackable=fallbackable
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ProviderRequestError(
            provider["name"],
            "خطای شبکه یا timeout",
            fallbackable=(provider["id"] == "cloudflare"),
        ) from None
    except (UnicodeError, json.JSONDecodeError):
        raise ProviderRequestError(
            provider["name"],
            "پاسخ JSON معتبر نبود",
            fallbackable=(provider["id"] == "cloudflare"),
        ) from None

    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProviderRequestError(
            provider["name"],
            "ساختار پاسخ معتبر نبود",
            fallbackable=(provider["id"] == "cloudflare"),
        ) from None

    return content.strip()


def request_chat_completion(messages, temperature=0.1):
    providers, notices = provider_chain()
    if not providers:
        detail = "؛ ".join(notices) if notices else "هیچ تنظیمی پیدا نشد"
        raise NoProviderAvailableError(
            "هیچ Provider قابل استفاده نیست. " + detail
        )

    failures = []
    for index, provider in enumerate(providers):
        try:
            content = _request_provider(provider, messages, temperature)
            return content, provider["name"]
        except ProviderRequestError as exc:
            failures.append(f"{exc.provider}: {exc.reason}")
            has_fallback = index + 1 < len(providers)
            if exc.fallbackable and has_fallback:
                print(
                    "⚠️ Cloudflare Workers AI در دسترس نیست؛ "
                    "تلاش امن با Gemini..."
                )
                continue
            break

    detail = "؛ ".join(failures + notices)
    raise NoProviderAvailableError(
        "هیچ Provider نتوانست درخواست را انجام دهد. " + detail
    )


def provider_status():
    providers, notices = provider_chain()
    by_id = {provider["id"]: provider for provider in providers}
    active = providers[0]["name"] if providers else "NONE"
    return {
        "active": active,
        "cloudflare": (
            f"configured ({CLOUDFLARE_MODEL})"
            if "cloudflare" in by_id else "not configured"
        ),
        "gemini": (
            "configured" if "gemini" in by_id else "not configured"
        ),
        "notices": notices,
    }


def extract_json_object(content):
    content = content.strip()
    if "```" in content:
        blocks = re.findall(
            r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE
        )
        if blocks:
            content = blocks[0].strip()
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            content = content[start:end + 1]
    return json.loads(content)


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
    if not isinstance(rel, str) or not rel.strip():
        raise Exception("مسیر فایل نامعتبر است.")

    target = (ROOT / rel).resolve()

    try:
        target.relative_to(ROOT)
    except ValueError:
        raise Exception(f"مسیر خارج از پروژه ممنوع است: {rel}")

    return target


def protected_path_reason(path):
    """Return a reason when a repository-relative path is sensitive/runtime."""
    raw = str(path).replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]

    if not raw or raw.startswith("/") or "\x00" in raw:
        return "مسیر نامعتبر"

    parts = [part.casefold() for part in PurePosixPath(raw).parts]
    if any(part == ".." for part in parts):
        return "مسیر خارج از پروژه"

    for part in parts[:-1]:
        if part in PROTECTED_DIR_NAMES:
            return f"runtime directory: {part}"

    name = parts[-1]
    if name == ".env" or name.startswith(".env."):
        return "environment file"
    if name in PROTECTED_FILE_NAMES:
        return "credential/runtime file"
    if any(name.endswith(suffix) for suffix in PROTECTED_SUFFIXES):
        return "credential/runtime file type"
    if re.search(
        r"(^|[._-])(api[._-]?keys?|tokens?|secrets?|credentials?)([._-]|$)",
        name,
        re.IGNORECASE,
    ):
        return "sensitive filename"

    return None


def should_ignore(path):
    return (
        any(part in DEFAULT_IGNORE for part in path.parts)
        or protected_path_reason(path) is not None
    )


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
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        content, provider_name = request_chat_completion(messages)
        result = extract_json_object(content)
        if not isinstance(result, dict):
            raise ValueError("AI response is not an object")
        result["_provider"] = provider_name
        return result
    except NoProviderAvailableError as exc:
        print(f"\n❌ {exc}")
        print(
            "Cloudflare: ~/.config/fox-ai/cloudflare.env | "
            "Gemini fallback: fox-ai config"
        )
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"\n❌ پاسخ Provider قابل پردازش نیست: {exc}")
        return None


def backup_file(path, timestamp):
    if not path.exists():
        return

    relative = path.relative_to(ROOT)
    reason = protected_path_reason(relative)
    if reason:
        raise Exception(
            f"فایل محافظت‌شده هرگز backup نمی‌شود: {relative}"
        )

    dest = ROOT / BACKUP_DIR / timestamp / relative
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
            reason = protected_path_reason(Path(rel))
            if reason:
                raise Exception(
                    f"تغییر فایل محافظت‌شده توسط AI ممنوع است: {rel}"
                )
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
    """Configure the existing Gemini fallback; Cloudflare stays env-only."""
    cfg = load_config()

    print("\nFox AI Configuration")
    print("Primary: Cloudflare Workers AI")
    print(f"  credentials: {CLOUDFLARE_ENV_FILE}")
    print("  required: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN")
    print("  security: chmod 600 ~/.config/fox-ai/cloudflare.env")
    print("\nFallback: Gemini (existing config)\n")

    base = input("Gemini Base URL [Enter = keep current]: ").strip()
    if base:
        cfg["base_url"] = base.rstrip("/")

    key = getpass.getpass("Gemini API Key [Enter = keep current]: ").strip()
    if key:
        cfg["api_key"] = key

    model = input(
        f"Gemini Model [{cfg.get('model')}]: "
    ).strip()
    if model:
        cfg["model"] = model

    save_config(cfg)
    print("\n✅ تنظیمات Gemini fallback ذخیره شد.")
    print(CONFIG_FILE)
    print("هیچ Cloudflare credential در config.json ذخیره نشد.")


def status():
    print("\n🦊 Fox AI")
    print("Project:", ROOT)
    print("Files:", len(collect_files()))

    state = provider_status()
    print("Active Provider:", state["active"])
    print("Cloudflare:", state["cloudflare"])
    print("Gemini fallback:", state["gemini"])
    for notice in state["notices"]:
        # Notices contain configuration state only, never credential values.
        if "ناقص" in notice or "نامعتبر" in notice or "ناامن" in notice:
            print("Provider warning:", notice)


def run_request(req):
    print("\n🦊 Fox AI")
    print("📂 Project:", ROOT)
    print("\n🧠 در حال بررسی پروژه...")

    result = ask_ai(req)

    if not result:
        return

    provider_name = result.pop("_provider", None)
    if provider_name:
        print(f"\n🔌 Provider: {provider_name}")
    print("\n🤖", result.get("summary", "No summary"))

    apply_actions(result)



def chat():
    print("""
🦊 Fox AI Chat
پروژه: {}
دستورهای داخلی:
  /exit       خروج
  /status     وضعیت پروژه و Git
  /git        وضعیت Git
  /diff       تغییرات Git
  /sync       دریافت امن تغییرات remote
  /push       add/commit/push با تأییدهای صریح
  /undo       برگشت آخرین backup

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
            git_status()
            continue

        if cmd == "/git":
            git_status()
            continue

        if cmd == "/diff":
            git_diff()
            continue

        if cmd == "/sync":
            git_sync()
            continue

        if cmd == "/push":
            git_push()
            continue

        if cmd == "/undo":
            undo()
            continue

        history.append({"role": "user", "content": user})

        context = project_context()
        chat_system = SYSTEM + r"""

این یک گفت‌وگوی چندمرحله‌ای است.
اگر کاربر درباره پروژه سؤال کرد، فایل‌های پروژه را بررسی کن.
اگر درخواست تغییر داد، actionهای دقیق تولید کن.
اگر کاربر گفت تست کن، تست مناسب پیشنهاد بده.
اگر درخواست فقط توضیح بود، actions را خالی بگذار.
"""
        messages = [{"role": "system", "content": chat_system}]

        # حفظ چند پیام اخیر برای جلوگیری از بزرگ شدن بیش از حد context
        messages.extend(history[-10:])
        messages.append({
            "role": "user",
            "content": f"""
PROJECT ROOT:
{ROOT}

CURRENT PROJECT FILES:
{context}

CURRENT REQUEST:
{user}
""",
        })

        try:
            print("\n🧠 در حال فکر کردن...")
            content, provider_name = request_chat_completion(messages)
            print(f"\n🔌 Provider: {provider_name}")
            print("\n🤖", content)

            try:
                result = extract_json_object(content)
                if isinstance(result, dict) and "actions" in result:
                    print("\n" + "=" * 60)
                    print("تغییرات پیشنهادی")
                    print("=" * 60)
                    apply_actions(result)
            except (json.JSONDecodeError, ValueError, TypeError):
                # پاسخ معمولی بوده و action قابل اجرا ندارد
                pass
        except NoProviderAvailableError as exc:
            print(f"\n❌ {exc}")
            print(
                "Cloudflare: ~/.config/fox-ai/cloudflare.env | "
                "Gemini fallback: fox-ai config"
            )



# --- Safe Git/GitHub integration ------------------------------------------


def _display_path(path):
    """Keep terminal control characters out of paths shown to the user."""
    return "".join(ch if ch.isprintable() else f"\\x{ord(ch):02x}" for ch in path)


def _redact_remote_url(url):
    # Redact URL user-info so embedded credentials are never displayed.
    return re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1***@", url)


def git_cmd(*args, root=None, timeout=60):
    """Run Git without a shell and return (returncode, stdout, stderr)."""
    worktree = Path(root or ROOT).resolve()
    if shutil.which("git") is None:
        return 127, "", "دستور git نصب نیست. در Termux اجرا کنید: pkg install git"

    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "1")
    try:
        result = subprocess.run(
            ["git", *[str(arg) for arg in args]],
            cwd=worktree,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return (
            result.returncode,
            result.stdout.rstrip("\n"),
            result.stderr.rstrip("\n"),
        )
    except subprocess.TimeoutExpired:
        return 124, "", "مهلت اجرای git تمام شد."
    except Exception as exc:
        return 1, "", str(exc)


def _git_interactive(*args, root=None):
    """Run an explicitly approved operation with Git attached to the terminal."""
    worktree = Path(root or ROOT).resolve()
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "1")
    try:
        result = subprocess.run(
            ["git", *[str(arg) for arg in args]],
            cwd=worktree,
            env=env,
        )
        return result.returncode
    except Exception as exc:
        print(f"❌ اجرای git ناموفق بود: {exc}")
        return 1


def _git_root(show_error=True):
    code, out, err = git_cmd("rev-parse", "--show-toplevel")
    if code != 0 or not out:
        if show_error:
            print("❌ پوشه فعلی داخل Git repository نیست.")
            if err:
                print(err)
        return None
    return Path(out).resolve()


def _remote_target(root, branch, remotes):
    """Find the configured remote and remote branch without guessing silently."""
    remote = ""
    remote_branch = branch

    if branch:
        _, configured_remote, _ = git_cmd(
            "config", "--get", f"branch.{branch}.remote", root=root
        )
        _, merge_ref, _ = git_cmd(
            "config", "--get", f"branch.{branch}.merge", root=root
        )
        if configured_remote in remotes and configured_remote != ".":
            remote = configured_remote
        if merge_ref.startswith("refs/heads/"):
            remote_branch = merge_ref[len("refs/heads/"):]

    if not remote:
        if "origin" in remotes:
            remote = "origin"
        elif len(remotes) == 1:
            remote = remotes[0]

    return remote, remote_branch


def _git_info(show_error=True):
    root = _git_root(show_error=show_error)
    if root is None:
        return None

    code, branch, _ = git_cmd(
        "symbolic-ref", "--quiet", "--short", "HEAD", root=root
    )
    if code != 0:
        branch = ""

    _, head, _ = git_cmd("rev-parse", "--short", "HEAD", root=root)
    _, remote_text, _ = git_cmd("remote", root=root)
    remotes = [item for item in remote_text.splitlines() if item]
    remote, remote_branch = _remote_target(root, branch, remotes)

    upstream = ""
    if branch:
        up_code, up_out, _ = git_cmd(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name",
            "@{upstream}", root=root,
        )
        if up_code == 0:
            upstream = up_out

    return {
        "root": root,
        "branch": branch,
        "head": head,
        "remotes": remotes,
        "remote": remote,
        "remote_branch": remote_branch,
        "upstream": upstream,
    }


def _status_entries(root):
    code, output, err = git_cmd(
        "status", "--porcelain=v1", "-z", "--untracked-files=all", root=root
    )
    if code != 0:
        return None, err or output

    records = output.split("\x00")
    entries = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 3:
            continue

        xy = record[:2]
        path = record[3:]
        original = None
        # In porcelain v1 -z format a rename/copy is: "XY new\0old\0".
        if (xy[0] in "RC" or xy[1] in "RC") and index < len(records):
            original = records[index] or None
            index += 1
        entries.append({"xy": xy, "path": path, "original": original})

    return entries, ""


def _entry_paths(entry):
    paths = [entry["path"]]
    if entry.get("original"):
        paths.append(entry["original"])
    return paths


def _ahead_behind(root, left, right):
    code, output, _ = git_cmd(
        "rev-list", "--left-right", "--count", f"{left}...{right}", root=root
    )
    if code != 0:
        return None, None
    try:
        left_count, right_count = output.split()
        return int(left_count), int(right_count)
    except (TypeError, ValueError):
        return None, None


def git_status():
    info = _git_info()
    if info is None:
        return False

    root = info["root"]
    print("\n🦊 Git status")
    print(f"Repository: {root}")
    if info["branch"]:
        print(f"Branch: {info['branch']} ({info['head']})")
    else:
        print(f"Branch: detached HEAD ({info['head'] or 'unknown'})")

    if info["remotes"]:
        print("Remotes:")
        for remote in info["remotes"]:
            _, url, _ = git_cmd("remote", "get-url", remote, root=root)
            suffix = "  ← selected" if remote == info["remote"] else ""
            print(f"  {remote}: {_redact_remote_url(url)}{suffix}")
    else:
        print("Remotes: (none)")

    print(f"Upstream: {info['upstream'] or '(not configured)'}")
    if info["upstream"]:
        ahead, behind = _ahead_behind(root, "HEAD", info["upstream"])
        if ahead is not None:
            print(f"Ahead/behind: {ahead}/{behind}")

    entries, err = _status_entries(root)
    if entries is None:
        print(f"❌ git status: {err}")
        return False

    print("Changes:")
    if not entries:
        print("  clean")
    for entry in entries:
        reason = next(
            (protected_path_reason(path) for path in _entry_paths(entry)
             if protected_path_reason(path)),
            None,
        )
        protected = f"  [محافظت‌شده: {reason}]" if reason else ""
        rename = ""
        if entry.get("original"):
            rename = f" <- {_display_path(entry['original'])}"
        print(
            f"  {entry['xy']} {_display_path(entry['path'])}{rename}{protected}"
        )
    return True


def _safe_diff_paths(entries, staged):
    paths = []
    blocked = []
    for entry in entries:
        xy = entry["xy"]
        changed = (xy[0] not in (" ", "?", "!")) if staged else (
            xy[1] not in (" ", "?", "!")
        )
        if not changed:
            continue
        entry_paths = _entry_paths(entry)
        reasons = [protected_path_reason(path) for path in entry_paths]
        if any(reasons):
            blocked.extend(entry_paths)
            continue
        paths.extend(entry_paths)
    return sorted(set(paths)), sorted(set(blocked))


def git_diff():
    info = _git_info()
    if info is None:
        return False
    root = info["root"]
    entries, err = _status_entries(root)
    if entries is None:
        print(f"❌ git diff: {err}")
        return False

    unstaged, blocked_worktree = _safe_diff_paths(entries, staged=False)
    staged, blocked_index = _safe_diff_paths(entries, staged=True)
    printed = False

    if unstaged:
        code, output, err = git_cmd(
            "diff", "--no-ext-diff", "--color=never", "--", *unstaged, root=root
        )
        if code != 0:
            print(f"❌ git diff: {err or output}")
            return False
        if output:
            print("\n--- تغییرات stage نشده ---")
            print(output)
            printed = True

    if staged:
        code, output, err = git_cmd(
            "diff", "--cached", "--no-ext-diff", "--color=never", "--",
            *staged, root=root,
        )
        if code != 0:
            print(f"❌ git diff --cached: {err or output}")
            return False
        if output:
            print("\n--- تغییرات stage شده ---")
            print(output)
            printed = True

    blocked = sorted(set(blocked_worktree + blocked_index))
    if blocked:
        print("\n🔒 محتوای فایل‌های محافظت‌شده نمایش داده نشد:")
        for path in blocked:
            print(f"  {_display_path(path)}")

    untracked = [
        entry["path"] for entry in entries
        if entry["xy"] == "??" and not protected_path_reason(entry["path"])
    ]
    if untracked:
        print("\nفایل‌های جدید (برای diff ابتدا باید stage شوند):")
        for path in untracked:
            print(f"  {_display_path(path)}")

    if not printed and not blocked and not untracked:
        print("✅ هیچ تغییر محلی وجود ندارد.")
    return True


def _chunks(items, size=100):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _unstage_paths(root, paths):
    ok = True
    for chunk in _chunks(sorted(set(paths))):
        code, out, err = git_cmd("restore", "--staged", "--", *chunk, root=root)
        if code != 0:
            code, out, err = git_cmd("reset", "-q", "HEAD", "--", *chunk, root=root)
        if code != 0:
            print(f"❌ خارج کردن فایل محافظت‌شده از stage ناموفق بود: {err or out}")
            ok = False
    return ok


_SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("GitHub token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    ("credential in URL", re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.I)),
)
_SECRET_ASSIGNMENT = re.compile(
    r'''(?ix)
    ["']?(?:[a-z0-9]+[_-])*
    (api[_-]?(?:key|token)|access[_-]?token|auth[_-]?token|
    client[_-]?secret|account[_-]?id|password|passwd|private[_-]?key|
    credentials?)["']?
    \s*[:=]\s*["']([^"'\r\n]{8,})["']
    '''
)
_SECRET_UNQUOTED_ASSIGNMENT = re.compile(
    r'''(?imx)^
    \s*["']?(?:[a-z0-9]+[_-])*
    (api[_-]?(?:key|token)|access[_-]?token|auth[_-]?token|
    client[_-]?secret|account[_-]?id|password|passwd|private[_-]?key|
    credentials?)["']?
    \s*[:=]\s*([A-Za-z0-9_./+@:=~-]{8,})\s*(?:[#;].*)?$
    '''
)
_PLACEHOLDER_WORDS = (
    "example", "sample", "placeholder", "your_", "your-", "changeme",
    "change_me", "dummy", "not-a-real", "redacted", "<", "${", "{", "$",
)


def _secret_labels(text):
    labels = []
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            labels.append(label)

    for assignment_pattern in (_SECRET_ASSIGNMENT, _SECRET_UNQUOTED_ASSIGNMENT):
        for match in assignment_pattern.finditer(text):
            value = match.group(2).strip().casefold()
            if value and not any(word in value for word in _PLACEHOLDER_WORDS):
                labels.append(f"literal {match.group(1)}")

    return sorted(set(labels))


def _read_git_blob(root, object_name):
    size_code, size_text, _ = git_cmd("cat-file", "-s", object_name, root=root)
    if size_code != 0:
        return None
    try:
        if int(size_text.strip()) > MAX_SECRET_SCAN:
            return None
    except ValueError:
        return None

    code, content, _ = git_cmd(
        "cat-file", "blob", object_name, root=root, timeout=120
    )
    if code != 0 or "\x00" in content:
        return None
    return content


def _cached_paths(root):
    code, output, err = git_cmd(
        "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR",
        root=root,
    )
    if code != 0:
        return None, err or output
    return [path for path in output.split("\x00") if path], ""


def _scan_staged_secrets(root):
    paths, err = _cached_paths(root)
    if paths is None:
        return None, err

    findings = []
    for path in paths:
        content = _read_git_blob(root, f":{path}")
        if content is None:
            continue
        for label in _secret_labels(content):
            findings.append((path, label))
    return findings, ""


def _stage_safe_changes(root):
    entries, err = _status_entries(root)
    if entries is None:
        print(f"❌ خواندن تغییرات ناموفق بود: {err}")
        return False

    protected = []
    safe = []
    staged_protected = []
    for entry in entries:
        paths = _entry_paths(entry)
        reasons = [protected_path_reason(path) for path in paths]
        if any(reasons):
            protected.extend((path, reason) for path, reason in zip(paths, reasons) if reason)
            if entry["xy"][0] not in (" ", "?", "!"):
                staged_protected.extend(paths)
            continue
        safe.extend(paths)

    if staged_protected and not _unstage_paths(root, staged_protected):
        return False

    for chunk in _chunks(sorted(set(safe))):
        code, out, err = git_cmd("add", "-A", "--", *chunk, root=root)
        if code != 0:
            print(f"❌ git add ناموفق بود: {err or out}")
            return False

    # Re-check the actual index. This also protects against files staged before
    # Fox AI started and against Git pathspec edge cases.
    code, output, err = git_cmd(
        "diff", "--cached", "--name-only", "-z", root=root
    )
    if code != 0:
        print(f"❌ بررسی stage ناموفق بود: {err or output}")
        return False
    indexed = [path for path in output.split("\x00") if path]
    forbidden_indexed = [path for path in indexed if protected_path_reason(path)]
    if forbidden_indexed:
        if not _unstage_paths(root, forbidden_indexed):
            return False
        protected.extend(
            (path, protected_path_reason(path)) for path in forbidden_indexed
        )

    if protected:
        print("\n🔒 فایل‌های زیر هرگز stage/commit نمی‌شوند:")
        shown = set()
        for path, reason in protected:
            if path in shown:
                continue
            shown.add(path)
            print(f"  {_display_path(path)} ({reason})")

    findings, scan_error = _scan_staged_secrets(root)
    if findings is None:
        print(f"❌ اسکن امنیتی stage ناموفق بود: {scan_error}")
        return False
    if findings:
        bad_paths = sorted({path for path, _ in findings})
        _unstage_paths(root, bad_paths)
        print("\n❌ secret احتمالی پیدا شد؛ فایل‌ها از stage خارج شدند:")
        for path, label in findings:
            print(f"  {_display_path(path)} ({label})")
        return False

    code, _, _ = git_cmd("diff", "--cached", "--quiet", root=root)
    if code == 0:
        print("✅ پس از حذف فایل‌های محافظت‌شده، تغییری برای commit نماند.")
        return False
    if code != 1:
        print("❌ بررسی تغییرات stage شده ناموفق بود.")
        return False
    return True


def _confirm_exact(prompt, expected):
    try:
        answer = input(prompt).strip().casefold()
    except (EOFError, KeyboardInterrupt):
        print("\n❎ لغو شد.")
        return False
    return answer == expected.casefold()


def _outgoing_commits(root, remote):
    code, output, err = git_cmd(
        "rev-list", "--reverse", "HEAD", "--not", f"--remotes={remote}", root=root
    )
    if code != 0:
        return None, err or output
    return [commit for commit in output.splitlines() if commit], ""


def _commit_paths(root, commit):
    code, output, err = git_cmd(
        "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z",
        commit, root=root,
    )
    if code != 0:
        return None, err or output
    return [path for path in output.split("\x00") if path], ""


def _audit_outgoing(root, commits):
    """Audit every commit that a push would introduce to the remote."""
    findings = []
    for commit in commits:
        msg_code, message, msg_err = git_cmd(
            "show", "-s", "--format=%B", commit, root=root
        )
        if msg_code != 0:
            return None, msg_err or message
        for label in _secret_labels(message):
            findings.append((commit[:10], "<commit message>", label))

        paths, err = _commit_paths(root, commit)
        if paths is None:
            return None, err
        for path in paths:
            reason = protected_path_reason(path)
            if reason:
                findings.append((commit[:10], path, reason))
                continue
            content = _read_git_blob(root, f"{commit}:{path}")
            if content is None:
                continue  # Deleted, binary, or too large.
            for label in _secret_labels(content):
                findings.append((commit[:10], path, label))
    return findings, ""


def _refresh_remote(info):
    remote = info["remote"]
    code, out, err = git_cmd("fetch", "--no-tags", remote, root=info["root"], timeout=180)
    if code != 0:
        print(f"❌ git fetch ناموفق بود: {err or out}")
        return False
    return True


def git_push():
    """Safely stage, commit, and push; commit and push require separate consent."""
    info = _git_info()
    if info is None:
        return False
    root = info["root"]
    branch = info["branch"]
    remote = info["remote"]
    remote_branch = info["remote_branch"]

    if not branch:
        print("❌ در detached HEAD امکان commit/push امن وجود ندارد.")
        return False
    if not remote:
        print("❌ remote مشخص نیست؛ ابتدا origin یا upstream شاخه را تنظیم کنید.")
        return False

    _, remote_url, _ = git_cmd("remote", "get-url", remote, root=root)
    print(f"\n🦊 Repository: {root}")
    print(f"Branch: {branch}")
    print(f"Remote: {remote} ({_redact_remote_url(remote_url)})")
    print(f"Remote branch: {remote_branch}")

    entries, err = _status_entries(root)
    if entries is None:
        print(f"❌ git status ناموفق بود: {err}")
        return False

    if entries:
        git_diff()
        print(
            "\nبرای stage کردن فقط فایل‌های امن، عبارت stage را دقیقاً وارد کنید."
        )
        if not _confirm_exact("تأیید git add [stage/cancel]: ", "stage"):
            print("❎ git add/commit لغو شد.")
            return False
        if not _stage_safe_changes(root):
            return False

        print("\n📋 diff نهاییِ stage شده:")
        git_diff()
        try:
            message = input("\n📝 پیام commit (خالی = لغو): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n❎ لغو شد.")
            return False
        if not message:
            print("❎ commit لغو شد.")
            return False
        message_findings = _secret_labels(message)
        if message_findings:
            print("❌ پیام commit شبیه secret/credential است و پذیرفته نشد.")
            return False

        print('برای ساخت commit عبارت "commit" را دقیقاً وارد کنید.')
        if not _confirm_exact("تأیید commit [commit/cancel]: ", "commit"):
            print("❎ commit لغو شد؛ تغییرات فقط stage شده‌اند.")
            return False

        # --no-verify prevents a hook from silently adding files after the
        # security audit. The resulting commit is audited again before push.
        code, out, err = git_cmd(
            "commit", "--no-verify", "-m", message, root=root, timeout=180
        )
        if code != 0:
            print(f"❌ git commit ناموفق بود: {err or out}")
            return False
        print(out or "✅ commit ساخته شد.")

    if not _refresh_remote(info):
        return False

    remote_ref = f"refs/remotes/{remote}/{remote_branch}"
    ref_code, _, _ = git_cmd("show-ref", "--verify", "--quiet", remote_ref, root=root)
    if ref_code == 0:
        ahead, behind = _ahead_behind(root, "HEAD", remote_ref)
        if behind:
            print(
                f"❌ شاخه محلی {behind} commit عقب است؛ ابتدا fox-ai git sync را اجرا کنید."
            )
            return False
        if ahead == 0:
            print("✅ commit جدیدی برای push وجود ندارد.")
            return True

    commits, commit_error = _outgoing_commits(root, remote)
    if commits is None:
        print(f"❌ بررسی commitهای خروجی ناموفق بود: {commit_error}")
        return False
    if not commits:
        print("✅ commit جدیدی برای push وجود ندارد.")
        return True

    findings, audit_error = _audit_outgoing(root, commits)
    if findings is None:
        print(f"❌ ممیزی امنیتی push ناموفق بود: {audit_error}")
        return False
    if findings:
        print("❌ push مسدود شد؛ commit خروجی شامل داده محافظت‌شده است:")
        for commit, path, reason in findings[:50]:
            print(f"  {commit}  {_display_path(path)} ({reason})")
        if len(findings) > 50:
            print(f"  ... و {len(findings) - 50} مورد دیگر")
        return False

    print(f"\n{len(commits)} commit به {remote}/{remote_branch} ارسال خواهد شد.")
    code, summary, _ = git_cmd(
        "log", "--no-walk", "--oneline", "--decorate", *commits[-10:], root=root
    )
    if code == 0 and summary:
        print(summary)
    print('برای push عبارت "push" را دقیقاً وارد کنید.')
    if not _confirm_exact("تأیید push [push/cancel]: ", "push"):
        print("❎ push لغو شد؛ commit محلی باقی ماند.")
        return False

    result = _git_interactive(
        "push", "--set-upstream", remote,
        f"{branch}:refs/heads/{remote_branch}", root=root,
    )
    if result != 0:
        print("❌ git push ناموفق بود؛ commit محلی حذف نشده است.")
        return False

    print("🚀 push با موفقیت انجام شد.")
    return True


def git_sync():
    """Fetch and fast-forward the current branch after an explicit preview."""
    info = _git_info()
    if info is None:
        return False
    root = info["root"]
    branch = info["branch"]
    remote = info["remote"]
    remote_branch = info["remote_branch"]

    if not branch:
        print("❌ در detached HEAD امکان sync امن وجود ندارد.")
        return False
    if not remote:
        print("❌ remote مشخص نیست؛ ابتدا origin یا upstream را تنظیم کنید.")
        return False

    entries, err = _status_entries(root)
    if entries is None:
        print(f"❌ git status ناموفق بود: {err}")
        return False
    if entries:
        print("❌ برای جلوگیری از overwrite، sync فقط با working tree کاملاً clean اجرا می‌شود:")
        for entry in entries:
            print(f"  {entry['xy']} {_display_path(entry['path'])}")
        return False

    print(f"🦊 Fetching {remote} ({remote_branch}) ...")
    if not _refresh_remote(info):
        return False

    remote_ref = f"refs/remotes/{remote}/{remote_branch}"
    code, _, _ = git_cmd("show-ref", "--verify", "--quiet", remote_ref, root=root)
    if code != 0:
        print(f"❌ شاخه {remote}/{remote_branch} روی remote پیدا نشد.")
        return False

    ahead, behind = _ahead_behind(root, "HEAD", remote_ref)
    if ahead is None:
        print("❌ مقایسه شاخه محلی و remote ناموفق بود.")
        return False
    if ahead and behind:
        print(
            f"❌ شاخه‌ها diverged هستند (ahead={ahead}, behind={behind})؛ "
            "sync خودکار انجام نشد."
        )
        return False
    if not behind:
        if ahead:
            print(f"✅ شاخه local به‌روز و {ahead} commit جلوتر است.")
        else:
            print("✅ پروژه با remote همگام است.")
        return True

    print(f"\n{behind} commit دریافت و با fast-forward اعمال می‌شود:")
    _, log_output, _ = git_cmd(
        "log", "--oneline", "--decorate", f"HEAD..{remote_ref}", root=root
    )
    if log_output:
        print(log_output)
    print('برای به‌روزرسانی فایل‌های محلی عبارت "sync" را دقیقاً وارد کنید.')
    if not _confirm_exact("تأیید sync [sync/cancel]: ", "sync"):
        print("❎ sync لغو شد؛ فقط fetch انجام شده است.")
        return False

    code, out, err = git_cmd(
        "merge", "--ff-only", remote_ref, root=root, timeout=180
    )
    if code != 0:
        print(f"❌ fast-forward ناموفق بود: {err or out}")
        return False
    print(out or "✅ sync با موفقیت انجام شد.")
    return True


def _git_usage():
    print("""
Git commands:
  fox-ai git status    نمایش repository، branch، remote و تغییرات
  fox-ai git diff      نمایش diff امن (بدون محتوای فایل‌های حساس)
  fox-ai git sync      fetch و fast-forward پس از تأیید
  fox-ai git push      git add/commit/push با تأییدهای صریح و اسکن امنیتی

Aliasها: fox-ai diff | sync | push
""")


def main():
    global ROOT
    ROOT = project_root()

    args = sys.argv[1:]
    if not args or args[0].lower() in ("-h", "--help", "help"):
        print("""
🦊 Fox AI Coding Agent

استفاده:
  fox-ai "درخواست شما"
  fox-ai chat
  fox-ai config
  fox-ai status
  fox-ai undo

Providerها:
  Cloudflare Workers AI (اصلی): ~/.config/fox-ai/cloudflare.env
  Gemini (fallback): fox-ai config

Git واقعی:
  fox-ai git status
  fox-ai git diff
  fox-ai git sync
  fox-ai git push

هیچ commit یا push بدون تأیید صریح انجام نمی‌شود.
""")
        return 0

    command = args[0].lower()

    if command == "config":
        configure()
        return 0
    if command == "status":
        status()
        print()
        return 0 if git_status() else 1
    if command == "undo":
        undo()
        return 0
    if command == "chat":
        chat()
        return 0

    if command == "git":
        subcommand = args[1].lower() if len(args) > 1 else "status"
        handlers = {
            "status": git_status,
            "diff": git_diff,
            "sync": git_sync,
            "push": git_push,
            "publish": git_push,
        }
        handler = handlers.get(subcommand)
        if handler is None:
            print(f"❌ زیر‌دستور Git ناشناخته است: {subcommand}")
            _git_usage()
            return 2
        return 0 if handler() else 1

    aliases = {
        "diff": git_diff,
        "sync": git_sync,
        "push": git_push,
    }
    if command in aliases:
        return 0 if aliases[command]() else 1

    run_request(" ".join(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
