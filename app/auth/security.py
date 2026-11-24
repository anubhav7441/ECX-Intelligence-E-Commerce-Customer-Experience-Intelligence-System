"""Password hashing, input validation, login throttling and a small per-user rate limiter."""
import hashlib
import re
import time
from collections import defaultdict, deque
from werkzeug.security import check_password_hash, generate_password_hash

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,24}$")
TYPO_DOMAINS = {"gmial.com": "gmail.com", "gmai.com": "gmail.com", "gmail.co": "gmail.com", "gmail.con": "gmail.com", "gamil.com": "gmail.com", "gmil.com": "gmail.com", "gmaill.com": "gmail.com", "gmail.in": "gmail.com", "yahho.com": "yahoo.com", "yaho.com": "yahoo.com", "yahoo.con": "yahoo.com", "hotmial.com": "hotmail.com", "hotmail.con": "hotmail.com", "outlok.com": "outlook.com", "outlook.con": "outlook.com"}
COMMON = {"password123", "1234567890", "qwerty12345", "password1234", "admin12345", "letmein1234", "iloveyou123"}


def hash_password(pw):
    try:
        return generate_password_hash(pw, method="scrypt")
    except (ValueError, AttributeError):          # OpenSSL without scrypt
        return generate_password_hash(pw, method="pbkdf2:sha256:600000")


def hash_code(text):
    """One-way hash of a short e-mail code (the code itself is never stored)."""
    return hashlib.sha256(text.encode()).hexdigest()


def verify_password(pw_hash, pw):
    try:
        return check_password_hash(pw_hash, pw)
    except Exception:
        return False


def email_error(email):
    """Return a message if the email address is not valid, else None. (Whether the mailbox really exists cannot be checked without sending an email.)"""
    email = (email or "").strip()
    if not email or len(email) > 254 or email.count("@") != 1 or not EMAIL_RE.match(email):
        return "Enter a valid email address, for example name@example.com."
    local, domain = email.rsplit("@", 1)
    domain = domain.lower()
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return "The part before @ cannot start or end with a dot or have two dots in a row."
    labels = domain.split(".")
    if any((not l) or l.startswith("-") or l.endswith("-") or len(l) > 63 for l in labels) or not re.fullmatch(r"[a-z]{2,24}", labels[-1]):
        return "The domain after @ is not valid."
    if domain in TYPO_DOMAINS:
        return f"Did you mean {local}@{TYPO_DOMAINS[domain]}? Please check the spelling."
    if domain == "gmail.com":
        base = local.split("+")[0]
        if not (6 <= len(base) <= 30) or not re.fullmatch(r"[A-Za-z0-9.]+", base) or base.isdigit():
            return "This is not a valid Gmail address. Gmail names have 6 to 30 letters, numbers or dots."
    return None


def validate_registration(email, name, password, confirm, min_len=10):
    errs = {}
    email = (email or "").strip()
    name = re.sub(r"\s+", " ", (name or "").strip())
    err = email_error(email)
    if err:
        errs["email"] = err
    if not (2 <= len(name) <= 60) or not re.fullmatch(r"[\w .'\-]+", name, re.UNICODE):
        errs["name"] = "Name must be 2-60 characters (letters, numbers, spaces, . ' -)."
    pw = password or ""
    if len(pw) < min_len:
        errs["password"] = f"Password must be at least {min_len} characters."
    elif len(pw) > 128:
        errs["password"] = "Password must be at most 128 characters."
    elif not (re.search(r"[A-Za-z]", pw) and re.search(r"\d", pw)):
        errs["password"] = "Password must contain at least one letter and one number."
    elif pw.lower() in COMMON or pw.lower() == email.lower().split("@")[0]:
        errs["password"] = "That password is too easy to guess."
    elif pw != (confirm or ""):
        errs["confirm"] = "Passwords do not match."
    return email.lower(), name, errs


_FAILS = defaultdict(list)


def lock_remaining(key, max_failures, lock_seconds):
    now = time.time()
    _FAILS[key] = [t for t in _FAILS[key] if now - t < lock_seconds]
    if len(_FAILS[key]) >= max_failures:
        return int(lock_seconds - (now - _FAILS[key][0]))
    return 0


def record_failure(key):
    _FAILS[key].append(time.time())


def clear_failures(key):
    _FAILS.pop(key, None)


_HITS = defaultdict(deque)


def rate_limited(key, per_minute):
    now = time.time()
    q = _HITS[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= per_minute:
        return True
    q.append(now)
    return False


def reset_state():
    _FAILS.clear(); _HITS.clear()
