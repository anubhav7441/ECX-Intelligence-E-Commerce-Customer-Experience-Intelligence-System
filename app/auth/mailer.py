"""Send the sign-up verification code by e-mail. Uses the same ECX_SMTP_* settings as the digest; credentials come only from
environment variables (.env), never from code. If they are not all set, e-mail verification is simply not available."""
import os
import smtplib
import ssl
from email.message import EmailMessage

NEEDED = ("ECX_SMTP_HOST", "ECX_SMTP_PORT", "ECX_SMTP_USER", "ECX_SMTP_PASSWORD", "ECX_SMTP_FROM")


def configured(env=None):
    env = os.environ if env is None else env
    return all(env.get(v) for v in NEEDED)


def send(to, subject, body, env=None):
    env = os.environ if env is None else env
    if not configured(env):
        raise RuntimeError("E-mail is not configured.")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, env["ECX_SMTP_FROM"], to
    msg.set_content(body)
    with smtplib.SMTP(env["ECX_SMTP_HOST"], int(env["ECX_SMTP_PORT"]), timeout=20) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(env["ECX_SMTP_USER"], env["ECX_SMTP_PASSWORD"])
        s.send_message(msg)
