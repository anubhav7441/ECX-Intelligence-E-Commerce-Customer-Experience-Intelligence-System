"""Structured (Pydantic) outputs for every agent step."""
from typing import Any, List, Optional
from pydantic import BaseModel


class ToolCall(BaseModel):
    name: str
    args: dict = {}
    ok: bool = True
    seconds: float = 0.0
    error: Optional[str] = None


class Finding(BaseModel):
    statement: str
    metric: Optional[str] = None
    value: Optional[Any] = None
    stat_result: Optional[dict] = None
    confidence: str = "Medium"          # High / Medium / Low
    limitation: str = ""
    evidence_id: Optional[str] = None


class AgentReport(BaseModel):
    agent: str
    task: str
    status: str = "ok"                  # ok | partial | failed
    data_sources: List[str] = []
    tool_calls: List[ToolCall] = []
    findings: List[Finding] = []
    uncertainty: List[str] = []
    errors: List[str] = []
    recommended_next_step: str = ""


class ProposedAction(BaseModel):
    action_id: Optional[str] = None
    title: str
    priority: str
    owner_role: str
    reason: str
    evidence_ids: List[str]


class InvestigationReport(BaseModel):
    run_id: str
    target_type: str
    target_id: str
    question: str
    plan: List[str]
    reports: List[AgentReport]
    verdicts: List[dict] = []
    actions: List[ProposedAction] = []
    summary: str
    overall_confidence: str
    limitations: List[str]
    human_review_required: bool = True
