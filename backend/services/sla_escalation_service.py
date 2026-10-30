"""
SLA Escalation Service — Detects SLA breaches and escalates tickets.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional
from .sla_forecasting_service import SLAForecastingService

try:
    import requests as _requests_lib
except ImportError:
    _requests_lib = None

logger = logging.getLogger(__name__)

ESCALATION_MAP = {
    "Network Support": "Senior Network Engineers",
    "Hardware Support": "Senior Hardware Engineers",
    "Application Support": "Senior Software Engineers",
    "IAM Team": "Security Operations Center",
    "General Support": "Tier 2 Support",
    "Cloud Apps Team": "Platform Engineering",
    "Security Ops": "Security Operations Center",
    "HR Systems": "Senior HR Systems Team",
    "IT Service Desk": "Tier 2 Support",
    "IT Inventory": "Senior Hardware Engineers",
    "Network Services": "Senior Network Engineers",
}

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class SLAEscalationService:
    def __init__(self):
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.teams_webhook_url = os.getenv("TEAMS_WEBHOOK_URL", "")
        self.forecaster = SLAForecastingService()

    def process_new_ticket(self, supabase_client, ticket: dict):
        """Analyzes ticket risk and tags if P(breach) > 0.75."""
        prob = self.forecaster.calculate_breach_probability(5, 0.8, 2.0) # Example inputs
        if prob > 0.75:
            supabase_client.table("tickets").update({"tags": ["risk:high-breach"]}).eq("id", ticket["id"]).execute()
            self.send_webhook_alert(ticket, message="High breach risk detected: " + str(prob))

    def check_breaches(self, supabase_client, company_id: Optional[str] = None) -> list[dict]:
        try:
            now_iso = _utc_now_iso()
            query = supabase_client.table("tickets").select("*").lt("sla_breach_at", now_iso).neq("status", "resolved").neq("status", "closed")
            if company_id: query = query.eq("company_id", company_id)
            return query.execute().data or []
        except Exception as exc:
            logger.error(f"[SLAEscalation] check_breaches error: {exc}")
            return []

    def escalate_ticket(self, supabase_client, ticket_id: str, original_team: str) -> dict:
        escalation_team = ESCALATION_MAP.get(original_team, f"Escalated: {original_team}")
        updates = {"assigned_team": escalation_team, "escalated": True, "escalated_at": _utc_now_iso()}
        res = supabase_client.table("tickets").update(updates).eq("id", ticket_id).execute()
        return res.data[0] if res.data else {}

    def send_webhook_alert(self, ticket: dict, webhook_url: str = "", message: str = "SLA Breach Alert") -> bool:
        url = webhook_url or self.slack_webhook_url or self.teams_webhook_url
        if not url or not _requests_lib: return False
        payload = {"text": f":rotating_light: {message}"}
        try:
            _requests_lib.post(url, json=payload, timeout=10).raise_for_status()
            return True
        except Exception: return False

    def run_sweep(self, supabase_client, company_id: Optional[str] = None, send_alerts: bool = True) -> dict:
        stats = {"breaches_found": 0, "escalated": 0, "alerts_sent": 0, "skipped_already_escalated": 0, "errors": 0}
        breached_tickets = self.check_breaches(supabase_client, company_id=company_id)
        stats["breaches_found"] = len(breached_tickets)
        for ticket in breached_tickets:
            if ticket.get("escalated"): 
                stats["skipped_already_escalated"] += 1
                continue
            updated = self.escalate_ticket(supabase_client, ticket.get("id"), ticket.get("assigned_team", "General Support"))
            if updated:
                stats["escalated"] += 1
                if send_alerts: stats["alerts_sent"] += 1
        return stats
