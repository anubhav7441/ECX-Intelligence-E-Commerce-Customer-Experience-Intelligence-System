# Administrator Guide

Create an admin with `python scripts\create_admin.py` (hidden password prompt). Log in normally; **Admin Panel** appears in the sidebar.

| Tab | What it does |
|---|---|
| Users | list users (no password hashes shown); make admin/user; disable/enable (confirmation asked). The last active administrator, and yourself, cannot be demoted or disabled. |
| Datasets | all datasets with owner e-mail and size; delete (confirmation). The built-in Olist dataset cannot be deleted. |
| Analysis activity | latest 100 analysis/report runs by user |
| Audit history | latest 200 security and admin events (logins, failures, uploads, role changes) |
| System log | tail of `logs/ecx.log` (errors and warnings) |
| Demo settings | max demo rows (50-2000), predictions and investigations per guest. Applies to **guests only**; registered users/admins and the Olist research analysis are never limited. |

Good practice: use a strong unique password, keep `database/ecx_auth.db` and `database/.secret_key` private, and back up `database/` and `data/datasets/` if the uploaded data matters.
