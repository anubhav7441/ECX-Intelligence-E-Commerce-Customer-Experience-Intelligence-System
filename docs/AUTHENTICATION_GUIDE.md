# Authentication Guide

## What is implemented (verified by `tests/test_auth_flow.py`)
- **Register** (name, email, password + confirm). Email format, name length, password rules (>= 10 characters, letters and a number, not a common password) and duplicate emails (case-insensitive) are checked on the server.
- **Passwords** are stored only as salted hashes (scrypt via Werkzeug; PBKDF2-SHA256 600k as fallback). Never in plain text, never in logs.
- **Login** with a generic error ("Invalid email or password") so accounts cannot be guessed. After 5 failures per IP+email the login is locked for 5 minutes. A dummy hash is checked for unknown emails to even out response time.
- **Sessions**: signed cookie, `HttpOnly`, `SameSite=Lax`, 8-hour lifetime, session reset at login (prevents fixation). Set `ECX_COOKIE_SECURE=1` when serving over HTTPS.
- **CSRF**: every POST/PUT/PATCH/DELETE needs the session token (hidden form field or `X-CSRF-Token` header; the front-end adds it automatically).
- **Logout** is a POST and clears the session.
- **Open-redirect protection** on the `next` parameter.
- **Roles**: `guest` (not logged in), `user`, `admin`. Disabled accounts cannot log in and are logged out on their next request.
- **Audit log**: register, login, failed login, logout, profile/password change, uploads, admin actions.

## Creating the first administrator (safe method)
```powershell
python scripts\create_admin.py
```
You type the email, name and a hidden password. Nothing is written to any file except the password **hash** in the database. There are no default or hardcoded accounts.

## Secret key
Not stored in source code. Order of use: environment variable `SECRET_KEY`, else `database/.secret_key` (random, created on first run, excluded from the ZIP). Do not share that file.

## Configuration (`config.py` / environment)
`ECX_AUTH_REQUIRED` (default 1), `ECX_AUTH_DB`, `ECX_COOKIE_SECURE`, `MIN_PASSWORD_LENGTH`, `LOGIN_MAX_FAILURES`, `LOGIN_LOCK_SECONDS`, `RATE_LIMIT_PER_MINUTE`, `DEMO_ROW_LIMIT`, `DEMO_MAX_PREDICTIONS`, `DEMO_MAX_INVESTIGATIONS`.

## Not implemented (be honest in your viva)
Email verification, "forgot password" by email, two-factor authentication, account deletion by the user, distributed (multi-server) rate limiting.
