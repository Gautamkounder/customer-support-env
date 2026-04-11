"""
Easy Grader — scores ticket classification accuracy.

Scoring (0.0 – 1.0):
  • Category match:  0.40
  • Priority match:  0.30
  • Sentiment match: 0.30

Partial credit for "close" answers:
  - Priority off by one level → half credit
  - Sentiment confusion between neutral↔negative → half credit
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..models import (
    CustomerSupportAction,
    CustomerTicket,
    TicketPriority,
    TicketSentiment,
)

PRIORITY_ORDER = [
    TicketPriority.LOW,
    TicketPriority.MEDIUM,
    TicketPriority.HIGH,
    TicketPriority.URGENT,
]

SENTIMENT_ADJACENCY = {
    (TicketSentiment.POSITIVE, TicketSentiment.NEUTRAL): 0.5,
    (TicketSentiment.NEUTRAL, TicketSentiment.POSITIVE): 0.5,
    (TicketSentiment.NEUTRAL, TicketSentiment.NEGATIVE): 0.5,
    (TicketSentiment.NEGATIVE, TicketSentiment.NEUTRAL): 0.5,
    (TicketSentiment.NEGATIVE, TicketSentiment.ANGRY): 0.5,
    (TicketSentiment.ANGRY, TicketSentiment.NEGATIVE): 0.5,
}


class EasyGrader:
    """Deterministic grader for easy (classification) task."""

    CATEGORY_WEIGHT = 0.40
    PRIORITY_WEIGHT = 0.30
    SENTIMENT_WEIGHT = 0.30

    @classmethod
    def grade(
        cls, action: CustomerSupportAction, ticket: CustomerTicket
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Returns (score, breakdown, feedback).
        """
        breakdown: Dict[str, float] = {}
        feedback_parts = []

        # ── Category ──
        if action.classify_category is None:
            breakdown["category"] = 0.0
            feedback_parts.append("❌ Category: not provided")
        elif action.classify_category == ticket.true_category:
            breakdown["category"] = 1.0
            feedback_parts.append(f"✅ Category: correct ({ticket.true_category.value})")
        else:
            breakdown["category"] = 0.0
            feedback_parts.append(
                f"❌ Category: {action.classify_category.value} "
                f"(expected {ticket.true_category.value})"
            )

        # ── Priority ──
        if action.classify_priority is None:
            breakdown["priority"] = 0.0
            feedback_parts.append("❌ Priority: not provided")
        elif action.classify_priority == ticket.true_priority:
            breakdown["priority"] = 1.0
            feedback_parts.append(f"✅ Priority: correct ({ticket.true_priority.value})")
        else:
            pred_idx = PRIORITY_ORDER.index(action.classify_priority)
            true_idx = PRIORITY_ORDER.index(ticket.true_priority)
            if abs(pred_idx - true_idx) == 1:
                breakdown["priority"] = 0.5
                feedback_parts.append(
                    f"⚠️ Priority: {action.classify_priority.value} "
                    f"(expected {ticket.true_priority.value}, partial credit)"
                )
            else:
                breakdown["priority"] = 0.0
                feedback_parts.append(
                    f"❌ Priority: {action.classify_priority.value} "
                    f"(expected {ticket.true_priority.value})"
                )

        # ── Sentiment ──
        if action.classify_sentiment is None:
            breakdown["sentiment"] = 0.0
            feedback_parts.append("❌ Sentiment: not provided")
        elif action.classify_sentiment == ticket.true_sentiment:
            breakdown["sentiment"] = 1.0
            feedback_parts.append(f"✅ Sentiment: correct ({ticket.true_sentiment.value})")
        else:
            pair = (action.classify_sentiment, ticket.true_sentiment)
            partial = SENTIMENT_ADJACENCY.get(pair, 0.0)
            breakdown["sentiment"] = partial
            if partial > 0:
                feedback_parts.append(
                    f"⚠️ Sentiment: {action.classify_sentiment.value} "
                    f"(expected {ticket.true_sentiment.value}, partial credit)"
                )
            else:
                feedback_parts.append(
                    f"❌ Sentiment: {action.classify_sentiment.value} "
                    f"(expected {ticket.true_sentiment.value})"
                )

        score = (
            breakdown["category"] * cls.CATEGORY_WEIGHT
            + breakdown["priority"] * cls.PRIORITY_WEIGHT
            + breakdown["sentiment"] * cls.SENTIMENT_WEIGHT
        )
        # OpenEnv strict requirement: (0, 1) exclusive
        score = max(0.0001, min(0.9999, score))
        breakdown = {k: max(0.0001, min(0.9999, float(v))) for k, v in breakdown.items()}
        
        feedback = "\n".join(feedback_parts)
        return round(score, 4), breakdown, feedback
