"""
Easy Task — Ticket Classification.

The agent receives a support ticket and must classify it by:
  • Category  (billing / technical / account / shipping / product / general)
  • Priority  (low / medium / high / urgent)
  • Sentiment (positive / neutral / negative / angry)

Single step.  Score is the average accuracy across the three fields.
"""

from __future__ import annotations

from typing import List

from ..models import CustomerTicket, TicketCategory, TicketPriority, TicketSentiment


# Tickets that are well-suited for the easy task (clear-cut classification)
EASY_TICKET_IDS = [
    "TKT-001", "TKT-002", "TKT-003", "TKT-005",
    "TKT-006", "TKT-009", "TKT-011", "TKT-014",
]

TASK_DESCRIPTION = (
    "You are a customer support triage agent. "
    "Read the following support ticket and classify it.\n\n"
    "You MUST provide:\n"
    "  1. classify_category — one of: billing, technical, account, shipping, product, general\n"
    "  2. classify_priority — one of: low, medium, high, urgent\n"
    "  3. classify_sentiment — one of: positive, neutral, negative, angry\n\n"
    "Respond with ONLY these three fields in your action."
)


class EasyTask:
    """Metadata & helpers for the easy (classification) task."""

    task_id = "easy_classify"
    name = "Ticket Classification"
    difficulty = "easy"
    max_steps = 1
    description = TASK_DESCRIPTION
    ticket_ids: List[str] = EASY_TICKET_IDS

    @staticmethod
    def strip_ground_truth(ticket: CustomerTicket) -> CustomerTicket:
        """Return a copy of the ticket with ground-truth fields removed."""
        return CustomerTicket(
            ticket_id=ticket.ticket_id,
            customer_name=ticket.customer_name,
            subject=ticket.subject,
            body=ticket.body,
            # ground truth hidden
            true_category=None,
            true_priority=None,
            true_sentiment=None,
            requires_escalation=None,
            expected_resolution_points=None,
        )
