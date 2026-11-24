"""Plain-English advice cards. Every card = Problem -> Why it is happening -> What to do, plus the numbers it is based on.
Rules fire ONLY when the data supports them; nothing is invented. 'Why' statements describe what the data shows (association),
never a proven cause unless a controlled test exists."""

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Good news": 3}


def confidence(n, strong_n=1000, ok_n=100):
    """Simple 3-level label based on how much data backs the statement."""
    return "Strong evidence" if n >= strong_n else ("Some evidence" if n >= ok_n else "Early signal (small sample)")


def card(key, title, problem, why, solutions, evidence, priority="Medium", n=None, area="General", tip=None):
    return {"key": key, "title": title, "problem": problem, "why": why, "solutions": list(solutions), "evidence": list(evidence),
            "priority": priority, "area": area, "confidence": confidence(n) if n is not None else "Based on summary figures", "n": n, "tip": tip}


def rank(cards):
    return sorted(cards, key=lambda c: (PRIORITY_ORDER.get(c["priority"], 9), -(c.get("n") or 0)))


def money(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"
