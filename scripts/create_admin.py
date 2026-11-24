"""Create (or promote) an administrator account from the command line.
Usage (from the project folder):   python scripts/create_admin.py
You are asked for the details; the password is typed hidden and is never stored in any file or in the code."""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app            # noqa: E402
from app.auth import db as adb, security as sec   # noqa: E402


def main():
    app = create_app()
    with app.app_context():
        email = input("Admin email: ").strip()
        name = input("Full name: ").strip()
        pw = getpass.getpass("Password (hidden): ")
        pw2 = getpass.getpass("Repeat password: ")
        e, n, errs = sec.validate_registration(email, name, pw, pw2, app.config["MIN_PASSWORD_LENGTH"])
        if errs:
            print("Cannot create the account:"); [print(" -", m) for m in errs.values()]; sys.exit(1)
        existing = adb.user_by_email(e)
        if existing:
            if input("That email already exists. Make it an administrator and reset its password? [y/N] ").lower() != "y":
                sys.exit("Nothing changed.")
            with adb.conn() as c:
                c.execute("UPDATE users SET role='admin', active=1, approved=1, password_hash=?, name=? WHERE id=?", (sec.hash_password(pw), n, existing["id"]))
        else:
            adb.create_user(e, n, sec.hash_password(pw), "admin")
        adb.audit("admin_created", f"via command line: {e}", None, "cli")
        print(f"Done. '{e}' can now log in as an administrator.")


if __name__ == "__main__":
    main()
