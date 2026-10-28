"""
audit.py — SQLite audit logging and human-review workflow.

Design decisions
----------------
- Parameterized queries throughout to prevent SQL injection.
- Only the minimum necessary information is stored.
- The original AI prediction is preserved even when a reviewer overrides it.
- Review actions are separate events linked by decision_id.
- No raw training records or personal data beyond what is needed are stored.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import DATABASE_PATH, MODEL_VERSION

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_DECISIONS_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id          TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    model_version        TEXT NOT NULL,
    input_snapshot       TEXT,          -- JSON of feature values (no PII)
    original_prediction  INTEGER NOT NULL,
    prediction_prob      REAL,
    risk_level           TEXT NOT NULL,
    risk_score           INTEGER NOT NULL,
    risk_reasons         TEXT,          -- JSON array of flag names
    explanation_summary  TEXT,
    privacy_passed       INTEGER,
    security_passed      INTEGER,
    reliability_passed   INTEGER,
    fairness_warning     INTEGER,
    routing_outcome      TEXT NOT NULL,
    routing_reason       TEXT,
    review_status        TEXT DEFAULT 'PENDING',  -- PENDING / REVIEWED / NA
    final_decision       INTEGER,       -- NULL until reviewed or auto-resolved
    reviewer_action      TEXT,          -- APPROVE / REJECT / REQUEST_REVIEW
    reviewer_id          TEXT,
    reviewer_comments    TEXT,
    reviewed_at          TEXT
);
"""

_CREATE_REVIEW_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS review_events (
    event_id        TEXT PRIMARY KEY,
    decision_id     TEXT NOT NULL REFERENCES decisions(decision_id),
    created_at      TEXT NOT NULL,
    reviewer_id     TEXT NOT NULL,
    action          TEXT NOT NULL,
    comments        TEXT,
    previous_status TEXT,
    new_status      TEXT
);
"""

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they do not already exist."""
    with _get_conn() as conn:
        conn.execute(_CREATE_DECISIONS_SQL)
        conn.execute(_CREATE_REVIEW_EVENTS_SQL)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def log_decision(
    input_snapshot: dict,
    original_prediction: int,
    prediction_prob: Optional[float],
    risk_level: str,
    risk_score: int,
    risk_flags: list,
    explanation_summary: str,
    privacy_passed: bool,
    security_passed: bool,
    reliability_passed: bool,
    fairness_warning: bool,
    routing_outcome: str,
    routing_reason: str,
    auto_resolve: bool = False,
) -> str:
    """
    Insert a new decision record and return the decision_id.

    For low-risk decisions that are auto-resolved, review_status is set to 'NA'
    and final_decision is set immediately.
    """
    import json

    decision_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    review_status  = "NA" if auto_resolve else "PENDING"
    final_decision = original_prediction if auto_resolve else None

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO decisions (
                decision_id, created_at, model_version, input_snapshot,
                original_prediction, prediction_prob,
                risk_level, risk_score, risk_reasons, explanation_summary,
                privacy_passed, security_passed, reliability_passed,
                fairness_warning, routing_outcome, routing_reason,
                review_status, final_decision
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                decision_id, now, MODEL_VERSION,
                json.dumps(input_snapshot),
                int(original_prediction),
                float(prediction_prob) if prediction_prob is not None else None,
                risk_level, int(risk_score),
                json.dumps([f["flag"] for f in risk_flags]),
                explanation_summary,
                int(privacy_passed), int(security_passed),
                int(reliability_passed), int(fairness_warning),
                routing_outcome, routing_reason,
                review_status, final_decision,
            ),
        )

    return decision_id


def record_review(
    decision_id: str,
    reviewer_id: str,
    action: str,           # "APPROVE" | "REJECT" | "REQUEST_REVIEW"
    comments: str = "",
) -> bool:
    """
    Record a human-reviewer decision.

    The original AI prediction is never modified.  The reviewer action and
    the resulting final_decision are stored separately.
    """
    import json

    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        # Fetch current decision
        row = conn.execute(
            "SELECT original_prediction, review_status FROM decisions "
            "WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()

        if row is None:
            return False

        previous_status = row["review_status"]

        # Determine final decision based on reviewer action
        if action == "APPROVE":
            final = row["original_prediction"]
            new_status = "REVIEWED"
        elif action == "REJECT":
            final = 1 - row["original_prediction"]   # flip the prediction
            new_status = "REVIEWED"
        else:  # REQUEST_REVIEW
            final = None
            new_status = "PENDING"

        conn.execute(
            """
            UPDATE decisions
            SET reviewer_action   = ?,
                reviewer_id       = ?,
                reviewer_comments = ?,
                reviewed_at       = ?,
                review_status     = ?,
                final_decision    = ?
            WHERE decision_id = ?
            """,
            (action, reviewer_id, comments, now, new_status, final, decision_id),
        )

        conn.execute(
            """
            INSERT INTO review_events
                (event_id, decision_id, created_at, reviewer_id,
                 action, comments, previous_status, new_status)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()), decision_id, now,
                reviewer_id, action, comments,
                previous_status, new_status,
            ),
        )

    return True


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_decision(decision_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single decision by ID."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_decisions(
    limit: int = 100,
    offset: int = 0,
    risk_level: Optional[str] = None,
    routing: Optional[str] = None,
    review_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch decisions with optional filters."""
    clauses = []
    params: list = []

    if risk_level:
        clauses.append("risk_level = ?")
        params.append(risk_level)
    if routing:
        clauses.append("routing_outcome = ?")
        params.append(routing)
    if review_status:
        clauses.append("review_status = ?")
        params.append(review_status)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]

    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM decisions {where} ORDER BY created_at DESC "
            f"LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_reviews() -> List[Dict[str, Any]]:
    """Return all decisions awaiting human review."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE review_status = 'PENDING' "
            "ORDER BY created_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> Dict[str, int]:
    """Return high-level counts for the dashboard."""
    with _get_conn() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        high    = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE risk_level='HIGH'"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE review_status='PENDING'"
        ).fetchone()[0]
        reviewed = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE review_status='REVIEWED'"
        ).fetchone()[0]
    return {
        "total_decisions": total,
        "high_risk":       high,
        "pending_review":  pending,
        "reviewed":        reviewed,
    }
