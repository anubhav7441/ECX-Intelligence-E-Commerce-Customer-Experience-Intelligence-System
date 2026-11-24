"""Authentication pages/APIs: register, login, logout, account settings, workspace and analysis history."""
import secrets
from urllib.parse import urlparse
from flask import Blueprint, current_app, g, jsonify, redirect, render_template, request, session, url_for
from datetime import datetime, timedelta, timezone
from app.auth import db as adb, mailer, security as sec

auth = Blueprint("auth", __name__)


def safe_next(target):
    if not target:
        return None
    u = urlparse(target)
    return target if (not u.netloc and not u.scheme and target.startswith("/") and not target.startswith("//") and "\\" not in target) else None


def ip():
    return request.remote_addr


def login_user(user):
    session.clear()                                   # prevents session fixation
    session["uid"] = user["id"]
    session["csrf"] = secrets.token_urlsafe(24)
    session.permanent = True


CODE_MINUTES, CODE_TRIES, RESEND_SECONDS = 15, 5, 60


def signup_mode():
    m = current_app.config.get("SIGNUP_MODE", "auto")
    if m == "auto":
        return "email" if mailer.configured() else "open"
    return m if m in ("open", "email", "admin") else "open"


def _code_hash(uid, code):
    return sec.hash_code(f"{uid}:{code}")


def send_code(user, force=False):
    """Create and e-mail a new 6-digit code. Returns False when a code was sent less than a minute ago (and force is not set)."""
    last = user.get("verify_sent")
    if last and not force and (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() < RESEND_SECONDS:
        return False
    code = f"{secrets.randbelow(10**6):06d}"
    mailer.send(user["email"], "Your ECX-Intelligence verification code",
                f"Hello {user['name']},\n\nYour verification code is {code}\nIt works for {CODE_MINUTES} minutes. If you did not sign up, ignore this e-mail.\n\nECX-Intelligence")
    exp = (datetime.now(timezone.utc) + timedelta(minutes=CODE_MINUTES)).isoformat(timespec="seconds")
    with adb.conn() as c:
        c.execute("UPDATE users SET verify_hash=?, verify_expires=?, verify_tries=0, verify_sent=? WHERE id=?", (_code_hash(user["id"], code), exp, adb.now(), user["id"]))
    return True


@auth.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("auth.workspace"))
    if request.method == "GET":
        return render_template("register.html", errors={}, form={})
    f = request.form
    email, name, errs = sec.validate_registration(f.get("email"), f.get("name"), f.get("password"), f.get("confirm"), current_app.config["MIN_PASSWORD_LENGTH"])
    if not errs and signup_mode() == "email" and sec.rate_limited(f"reg|{ip()}", 5):      # stops the form being used to flood inboxes with codes
        return render_template("register.html", errors={"email": "Too many sign-up attempts. Please wait a minute."}, form={"email": f.get("email", ""), "name": f.get("name", "")}), 429
    if not errs and signup_mode() == "email":
        old = adb.user_by_email(email)
        if old and not old.get("approved", 1) and old["role"] == "user":               # never-verified sign-up: the real owner of the address may take it over
            with adb.conn() as c:
                c.execute("DELETE FROM users WHERE id=? AND approved=0", (old["id"],))
    if not errs and adb.user_by_email(email):
        errs["email"] = "An account with this email already exists. Try logging in."
    if errs:
        return render_template("register.html", errors=errs, form={"email": f.get("email", ""), "name": f.get("name", "")}), 400
    try:
        mode = signup_mode()
        uid = adb.create_user(email, name, sec.hash_password(f["password"]), "user", approved=1 if mode == "open" else 0)
    except Exception:                                 # race: duplicate inserted between check and insert
        return render_template("register.html", errors={"email": "An account with this email already exists."}, form={"email": email, "name": name}), 400
    user = adb.user_by_id(uid)
    if mode == "email":                               # no login until the person types the code we e-mail to them
        try:
            send_code(user, force=True)
        except Exception:
            current_app.logger.exception("verification e-mail failed")
            with adb.conn() as c:
                c.execute("DELETE FROM users WHERE id=?", (uid,))
            return render_template("register.html", errors={"email": "We could not send the verification e-mail. Check the address and try again."}, form={"email": email, "name": name}), 503
        adb.audit("register_pending", "verification code e-mailed", user, ip())
        session["verify_email"] = email
        return redirect(url_for("auth.verify"))
    if mode == "admin":                               # no login until an administrator approves the account
        adb.audit("register_pending", "waiting for approval", user, ip())
        return redirect(url_for("auth.login", pending=1))
    adb.audit("register", "new account", user, ip())
    login_user(user)
    return redirect(url_for("auth.workspace"))


@auth.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("auth.workspace"))
    nxt = safe_next(request.values.get("next"))
    if request.method == "GET":
        note = "Thanks for signing up. An administrator must approve your account before you can sign in. Please check back later." if request.args.get("pending") else None
        return render_template("login.html", error=None, notice=note, next=nxt, email="")
    email = (request.form.get("email") or "").strip().lower()
    key = f"{ip()}|{email}"
    cfg = current_app.config
    wait = sec.lock_remaining(key, cfg["LOGIN_MAX_FAILURES"], cfg["LOGIN_LOCK_SECONDS"])
    if wait:
        return render_template("login.html", error=f"Too many failed attempts. Try again in {wait // 60 + 1} minute(s).", next=nxt, email=email), 429
    user = adb.user_by_email(email) if email else None
    # verify against a dummy hash when the user does not exist so timing does not reveal accounts
    ok = sec.verify_password(user["password_hash"] if user else _DUMMY, request.form.get("password") or "") and user is not None
    if not ok or not user["active"]:
        sec.record_failure(key)
        adb.audit("login_failed", f"email={email[:80]}", None, ip())
        msg = "This account has been disabled. Contact an administrator." if (ok is False and user and not user["active"] and sec.verify_password(user["password_hash"], request.form.get("password") or "")) else "Invalid email or password."
        return render_template("login.html", error=msg, next=nxt, email=email), 401
    if not user.get("approved", 1) and signup_mode() == "email":      # correct password but e-mail not confirmed yet: go and finish it
        session["verify_email"] = email
        try:
            send_code(user)
        except Exception:
            current_app.logger.exception("verification e-mail failed")
        return redirect(url_for("auth.verify"))
    if not user.get("approved", 1):                   # correct password, but not approved yet (shown only after the password is right)
        adb.audit("login_blocked_pending", "", user, ip())
        return render_template("login.html", error="Your account is waiting for administrator approval. You can sign in once it is approved.", next=nxt, email=email), 403
    sec.clear_failures(key)
    with adb.conn() as c:
        c.execute("UPDATE users SET last_login=? WHERE id=?", (adb.now(), user["id"]))
    login_user(user)
    adb.audit("login", "", user, ip())
    if user["role"] == "admin" and not nxt:           # administrators manage the system; they land on the admin page
        return redirect(current_app.config["ADMIN_PATH"])
    return redirect(nxt or url_for("auth.workspace"))


def _pending_user():
    e = session.get("verify_email")
    u = adb.user_by_email(e) if e else None
    return u if (u and u["active"] and not u.get("approved", 1)) else None


@auth.route("/verify", methods=["GET", "POST"])
def verify():
    u = _pending_user()
    if not u:
        session.pop("verify_email", None)
        return redirect(url_for("auth.login"))
    masked = u["email"][0] + "***@" + u["email"].split("@")[1]
    if request.method == "GET":
        return render_template("verify.html", masked=masked, error=None, notice=request.args.get("sent") and "A new code has been sent." or None)
    code = "".join(ch for ch in (request.form.get("code") or "") if ch.isdigit())
    err = None
    exp = u.get("verify_expires")
    if not u.get("verify_hash"):
        err = "No code is active. Press 'Send a new code'."
    elif u["verify_tries"] >= CODE_TRIES:
        err = "Too many wrong attempts. Press 'Send a new code'."
    elif exp and datetime.fromisoformat(exp) < datetime.now(timezone.utc):
        err = "This code has expired. Press 'Send a new code'."
    elif len(code) != 6 or not secrets.compare_digest(_code_hash(u["id"], code), u["verify_hash"]):
        with adb.conn() as c:
            c.execute("UPDATE users SET verify_tries=verify_tries+1 WHERE id=?", (u["id"],))
        err = "That code is not correct."
    if err:
        adb.audit("verify_failed", "", u, ip())
        return render_template("verify.html", masked=masked, error=err, notice=None), 400
    with adb.conn() as c:
        c.execute("UPDATE users SET approved=1, verify_hash=NULL, verify_expires=NULL, verify_tries=0, last_login=? WHERE id=?", (adb.now(), u["id"]))
    adb.audit("register", "e-mail verified", u, ip())
    login_user(adb.user_by_id(u["id"]))
    return redirect(url_for("auth.workspace"))


@auth.post("/verify/resend")
def verify_resend():
    u = _pending_user()
    if not u:
        return redirect(url_for("auth.login"))
    masked = u["email"][0] + "***@" + u["email"].split("@")[1]
    try:
        if not send_code(u):
            return render_template("verify.html", masked=masked, error="Please wait a minute before asking for another code.", notice=None), 429
    except Exception:
        current_app.logger.exception("verification e-mail failed")
        return render_template("verify.html", masked=masked, error="We could not send the e-mail. Try again later.", notice=None), 503
    return redirect(url_for("auth.verify", sent=1))


_DUMMY = sec.hash_password("not-a-real-password-0")


@auth.post("/logout")
def logout():
    if g.user:
        adb.audit("logout", "", g.user, ip())
    session.clear()
    return redirect(url_for("landing"))


@auth.route("/workspace")
def workspace():
    from multids import registry as R
    u = g.user
    runs = adb.list_runs(u["id"], 200) if u else []
    owned = [d for d in R.list_all() if u and d.get("owner_id") == u["id"] and d["dataset_id"] != "olist"]
    stats = {"datasets": len(owned), "analyses": sum(1 for r in runs if r["kind"] == "analysis"), "reports": sum(1 for r in runs if r["kind"] == "report")}
    note = []
    if runs and runs[0]["kind"] == "report":
        note.append(f"Your report for '{runs[0]['dataset_name']}' is ready ({runs[0]['ts'][:16].replace('T', ' ')}).")
    if not owned:
        note.append("You have not uploaded a dataset yet. Open Dataset Center to add a CSV or Excel file.")
    top, source = _top_advice(owned)
    return render_template("workspace.html", active="workspace", stats=stats, top=top, source=source, recent=runs[:6], reports=[r for r in runs if r["kind"] == "report"][:5],
                           datasets=owned[:6], notifications=note)


def _top_advice(owned):
    """The 3 most important pieces of advice: from the user's newest dataset if it is ready, otherwise from the Olist sample data."""
    try:
        from app.insights import dataset, olist
        for d in sorted(owned, key=lambda x: x.get("created_at") or "", reverse=True)[:1]:
            r = dataset.build(d["dataset_id"])
            if r.get("cards"):
                return [c for c in r["cards"] if c["priority"] != "Good news"][:6], d["name"]
        return [c for c in olist.build() if c["priority"] != "Good news"][:6], "Olist sample data"
    except Exception:
        current_app.logger.exception("could not build top advice")
        return [], None


@auth.route("/history")
def history():
    return render_template("history.html", active="history")


@auth.get("/api/history")
def api_history():
    u = g.user
    runs = adb.list_runs(u["id"] if u else None, 300)
    return jsonify(runs=runs)


@auth.route("/account")
def account():
    return render_template("account.html", active="account")


@auth.post("/api/account/profile")
def api_profile():
    b = request.get_json(silent=True) or {}
    name = " ".join(str(b.get("name", "")).split())
    if not (2 <= len(name) <= 60):
        return jsonify(error="Name must be 2-60 characters."), 400
    with adb.conn() as c:
        c.execute("UPDATE users SET name=? WHERE id=?", (name, g.user["id"]))
    adb.audit("profile_update", "", g.user, ip())
    return jsonify(ok=True, name=name)


@auth.post("/api/account/password")
def api_password():
    b = request.get_json(silent=True) or {}
    if not sec.verify_password(g.user["password_hash"], str(b.get("current", ""))):
        return jsonify(error="Current password is incorrect."), 400
    _, _, errs = sec.validate_registration(g.user["email"], g.user["name"], b.get("new"), b.get("confirm"), current_app.config["MIN_PASSWORD_LENGTH"])
    errs.pop("email", None); errs.pop("name", None)
    if errs:
        return jsonify(error=" ".join(errs.values())), 400
    with adb.conn() as c:
        c.execute("UPDATE users SET password_hash=? WHERE id=?", (sec.hash_password(b["new"]), g.user["id"]))
    adb.audit("password_change", "", g.user, ip())
    return jsonify(ok=True)
