"""Living metric-drag ledger: record issues that pull PR/ROC down."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "results" / "axion_gap_close" / "metric_drag_ledger.jsonl"
DEFAULT_REPORT = ROOT / "results" / "axion_gap_close" / "metric_drag_report.md"

DRAG_DECISIONS = {
    "ERROR",
    "GUARD_REGRESS",
    "REJECT",
    "FROZEN_DRIFT",
    "OOM",
    "TIMEOUT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_drag(
    *,
    track: str,
    dataset: str,
    setting: str = "semi-supervised",
    knobset: str = "",
    pr: Optional[float] = None,
    roc: Optional[float] = None,
    lock: Optional[float] = None,
    delta: Optional[float] = None,
    decision: str = "",
    error: Optional[str] = None,
    suspected_cause: str = "",
    paper_ids: Optional[List[str]] = None,
    action: str = "investigate",
    ledger_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one drag row. Never promotes; ledger is for diagnosis only."""
    path = Path(ledger_path or DEFAULT_LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    if delta is None and pr is not None and lock is not None:
        try:
            delta = float(pr) - float(lock)
        except (TypeError, ValueError):
            delta = None
    row: Dict[str, Any] = {
        "time": utc_now(),
        "track": track,
        "dataset": dataset,
        "setting": setting,
        "knobset": knobset,
        "PR": pr,
        "ROC": roc,
        "lock": lock,
        "delta": delta,
        "decision": decision,
        "error": error,
        "suspected_cause": suspected_cause,
        "paper_ids": paper_ids or [],
        "action": action,
    }
    if extra:
        row["extra"] = extra
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def should_record_drag(decision: str, delta: Optional[float], error: Optional[str]) -> bool:
    if error:
        return True
    if decision in DRAG_DECISIONS:
        return True
    if delta is not None and delta <= -0.5:
        return True
    return False


def suspected_cause_for(
    *,
    decision: str,
    dataset: str,
    error: Optional[str],
    delta: Optional[float],
) -> str:
    err = (error or "").lower()
    if "out of memory" in err or "cuda" in err and "memory" in err:
        return "cuda_oom"
    if "killed" in err or "memoryerror" in err:
        return "host_ram_oom"
    if decision == "GUARD_REGRESS":
        return f"guard_regress_{dataset}"
    if decision == "FROZEN_DRIFT":
        return "glass_frozen_drift"
    if decision == "ERROR":
        return "runtime_error"
    if decision == "REJECT" and delta is not None and delta < 0:
        return "knob_regress_vs_lock"
    if decision == "REJECT":
        return "knob_below_promote_delta"
    if delta is not None and delta <= -0.5:
        return "pr_delta_le_minus_0_5"
    return "metric_drag"


def rewrite_report(ledger_path: Optional[Path] = None, report_path: Optional[Path] = None) -> Path:
    """Roll up ledger into a short markdown report (top unpaid / frequent causes)."""
    ledger = Path(ledger_path or DEFAULT_LEDGER)
    report = Path(report_path or DEFAULT_REPORT)
    rows: List[Dict[str, Any]] = []
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_cause: Dict[str, int] = {}
    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        c = str(r.get("suspected_cause") or "unknown")
        by_cause[c] = by_cause.get(c, 0) + 1
        ds = str(r.get("dataset") or "?")
        by_ds.setdefault(ds, []).append(r)

    # Worst unpaid: most negative delta per dataset (last occurrence)
    unpaid: List[tuple] = []
    for ds, lst in by_ds.items():
        best = None
        for r in lst:
            d = r.get("delta")
            if d is None:
                continue
            try:
                d = float(d)
            except (TypeError, ValueError):
                continue
            if best is None or d < best[0]:
                best = (d, r)
        if best is not None:
            unpaid.append((best[0], ds, best[1]))
    unpaid.sort(key=lambda t: t[0])

    lines = [
        "# Metric drag report",
        "",
        f"Updated: {utc_now()}",
        f"Ledger rows: **{len(rows)}**",
        "",
        "## Top suspected causes",
        "",
        "| Cause | Count |",
        "|---|---|",
    ]
    for c, n in sorted(by_cause.items(), key=lambda kv: -kv[1])[:20]:
        lines.append(f"| {c} | {n} |")
    lines.extend(["", "## Worst PR deltas (by dataset)", "", "| Dataset | Delta | Decision | Track | Cause |", "|---|---|---|---|---|"])
    for d, ds, r in unpaid[:25]:
        lines.append(
            f"| {ds} | {d:+.2f} | {r.get('decision')} | {r.get('track')} | {r.get('suspected_cause')} |"
        )
    lines.extend(
        [
            "",
            "## Action",
            "",
            "- Merge with `pr_budget.md` / `known_bad_metrics.json` before promote.",
            "- Never promote rows from this ledger.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
