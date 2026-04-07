"""
Medium Task — Draft a Reply.

The agent receives a support ticket and must:
  1. Classify it (category, priority, sentiment)
  2. Draft a professional reply that addresses the customer's issues

Two-step task:
  Step 1 — Classification (same as easy)
  Step 2 — Draft reply + optional internal notes

Graded on classification accuracy + reply quality.
"""

from __future__ import annotations

from typing import List

from ..models import CustomerTicket


MEDIUM_TICKET_IDS = [
    "TKT-002", "TKT-004", "TKT-005", "TKT-006",
    "TKT-008", "TKT-009", "TKT-013",
]

TASK_DESCRIPTION_STEP1 = (
    "You are a customer support agent handling a support ticket.\n\n"
    "STEP 1 of 2: Classify this ticket.\n"
    "Provide classify_category, classify_priority, and classify_sentiment."
)

TASK_DESCRIPTION_STEP2 = (
    "STEP 2 of 2: Now draft a professional reply to the customer.\n\n"
    "Your reply should:\n"
    "  • Be empathetic and professional\n"
    "  • Address ALL issues raised in the ticket\n"
    "  • Provide concrete next steps\n"
    "  • Be appropriately detailed (not too short, not too long)\n\n"
    "Provide your response in the draft_reply field.\n"
    "Optionally, add internal_notes for your team."
)


class MediumTask:
    """Metadata & helpers for the medium (classify + reply) task."""

    task_id = "medium_reply"
    name = "Draft a Reply"
    difficulty = "medium"
    max_steps = 2
    description_step1 = TASK_DESCRIPTION_STEP1
    description_step2 = TASK_DESCRIPTION_STEP2
    ticket_ids: List[str] = MEDIUM_TICKET_IDS

    @staticmethod
    def get_description(step: int) -> str:
        if step == 0:
            return TASK_DESCRIPTION_STEP1
        return TASK_DESCRIPTION_STEP2

    @staticmethod
    def strip_ground_truth(ticket: CustomerTicket) -> CustomerTicket:
        return CustomerTicket(
            ticket_id=ticket.ticket_id,
            customer_name=ticket.customer_name,
            subject=ticket.subject,
            body=ticket.body,
        )
