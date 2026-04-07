"""
Hard Task — Full Resolution (Triage + Reply + Escalation).

The agent receives a complex, multi-issue support ticket and must:
  1. Classify it
  2. Draft a comprehensive reply
  3. Make correct escalation decisions

Three-step task with the most demanding grading criteria.
"""

from __future__ import annotations

from typing import List

from ..models import CustomerTicket


HARD_TICKET_IDS = [
    "TKT-004", "TKT-007", "TKT-010", "TKT-012",
    "TKT-013", "TKT-015",
]

TASK_DESCRIPTION_STEP1 = (
    "You are a senior customer support agent handling a complex ticket.\n\n"
    "STEP 1 of 3: Classify this ticket.\n"
    "Provide classify_category, classify_priority, and classify_sentiment."
)

TASK_DESCRIPTION_STEP2 = (
    "STEP 2 of 3: Draft a professional, empathetic reply to the customer.\n\n"
    "This is a complex ticket. Your reply must:\n"
    "  • Address EVERY issue or question raised\n"
    "  • Be sensitive to the customer's emotional state\n"
    "  • Provide specific, actionable next steps\n"
    "  • Maintain a professional tone even if the customer is angry\n"
    "  • Be thorough but not verbose\n\n"
    "Provide your response in the draft_reply field."
)

TASK_DESCRIPTION_STEP3 = (
    "STEP 3 of 3: Make your escalation decision.\n\n"
    "Decide whether this ticket needs escalation:\n"
    "  • escalation_level: none / supervisor / manager / legal\n"
    "  • escalation_reason: explain WHY you chose to escalate (or not)\n"
    "  • internal_notes: any notes for the team\n\n"
    "Consider: legal implications, customer risk level, issue severity, "
    "and whether the issue requires authority beyond a support agent."
)


class HardTask:
    """Metadata & helpers for the hard (full resolution) task."""

    task_id = "hard_resolution"
    name = "Full Resolution"
    difficulty = "hard"
    max_steps = 3
    ticket_ids: List[str] = HARD_TICKET_IDS

    @staticmethod
    def get_description(step: int) -> str:
        if step == 0:
            return TASK_DESCRIPTION_STEP1
        elif step == 1:
            return TASK_DESCRIPTION_STEP2
        return TASK_DESCRIPTION_STEP3

    @staticmethod
    def strip_ground_truth(ticket: CustomerTicket) -> CustomerTicket:
        return CustomerTicket(
            ticket_id=ticket.ticket_id,
            customer_name=ticket.customer_name,
            subject=ticket.subject,
            body=ticket.body,
        )
