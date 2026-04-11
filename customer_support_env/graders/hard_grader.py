"""
Hard Grader — scores classification + reply + escalation decision.

Overall scoring:
  • Classification accuracy:   0.20
  • Reply quality:             0.45
  • Escalation decision:       0.35
    - Correct escalation yes/no: 0.15
    - Appropriate level:         0.10
    - Reason quality:            0.10
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ..models import (
    CustomerSupportAction,
    CustomerTicket,
    EscalationLevel,
)
from .easy_grader import EasyGrader
from .medium_grader import MediumGrader


class HardGrader:
    """Grader for hard (full resolution) task."""

    CLASSIFICATION_WEIGHT = 0.20
    REPLY_WEIGHT = 0.45
    ESCALATION_WEIGHT = 0.35

    @classmethod
    def grade_escalation(
        cls,
        action: Optional[CustomerSupportAction],
        ticket: CustomerTicket,
    ) -> Tuple[float, Dict[str, float], str]:
        breakdown: Dict[str, float] = {}
        feedback_parts = []

        if action is None:
            return 0.0, {"escalation_decision": 0.0}, "❌ No escalation action provided"

        requires = ticket.requires_escalation or False
        agent_escalates = (
            action.escalation_level is not None
            and action.escalation_level != EscalationLevel.NONE
        )

        # ── Correct decision ──
        if requires == agent_escalates:
            breakdown["correct_decision"] = 1.0
            feedback_parts.append(
                f"✅ Escalation decision correct "
                f"({'escalated' if agent_escalates else 'not escalated'})"
            )
        else:
            breakdown["correct_decision"] = 0.0
            if requires and not agent_escalates:
                feedback_parts.append(
                    "❌ Should have escalated this ticket but didn't"
                )
            else:
                feedback_parts.append(
                    "❌ Escalated unnecessarily"
                )

        # ── Appropriate level ──
        if agent_escalates and requires:
            # Determine expected level based on ticket content
            expected_level = cls._infer_expected_level(ticket)
            if action.escalation_level == expected_level:
                breakdown["appropriate_level"] = 1.0
                feedback_parts.append(
                    f"✅ Escalation level correct ({expected_level.value})"
                )
            elif cls._is_adjacent_level(action.escalation_level, expected_level):
                breakdown["appropriate_level"] = 0.5
                feedback_parts.append(
                    f"⚠️ Escalation level close "
                    f"({action.escalation_level.value}, expected {expected_level.value})"
                )
            else:
                breakdown["appropriate_level"] = 0.0
                feedback_parts.append(
                    f"❌ Wrong escalation level "
                    f"({action.escalation_level.value}, expected {expected_level.value})"
                )
        elif not requires and not agent_escalates:
            breakdown["appropriate_level"] = 1.0  # Correctly didn't escalate
        else:
            breakdown["appropriate_level"] = 0.0

        # ── Reason quality ──
        if action.escalation_reason and len(action.escalation_reason.strip()) >= 20:
            # Check if reason contains relevant keywords
            reason_lower = action.escalation_reason.lower()
            relevant_keywords = cls._get_escalation_keywords(ticket)
            keyword_hits = sum(
                1 for kw in relevant_keywords if kw in reason_lower
            )
            keyword_ratio = keyword_hits / max(len(relevant_keywords), 1)
            breakdown["reason_quality"] = min(1.0, 0.5 + keyword_ratio * 0.5)
            if keyword_ratio >= 0.3:
                feedback_parts.append(
                    f"✅ Escalation reason is well-justified ({keyword_hits} relevant points)"
                )
            else:
                feedback_parts.append(
                    f"⚠️ Escalation reason could be more specific"
                )
        elif action.escalation_reason:
            breakdown["reason_quality"] = 0.3
            feedback_parts.append("⚠️ Escalation reason too brief")
        else:
            breakdown["reason_quality"] = 0.0 if requires else 0.8
            if requires:
                feedback_parts.append("❌ No escalation reason provided")
            else:
                feedback_parts.append("✅ No escalation reason needed")

        weights = {
            "correct_decision": 0.15 / cls.ESCALATION_WEIGHT,
            "appropriate_level": 0.10 / cls.ESCALATION_WEIGHT,
            "reason_quality": 0.10 / cls.ESCALATION_WEIGHT,
        }
        w_total = sum(weights.values())
        score = sum(
            breakdown.get(k, 0) * weights[k] / w_total
            for k in weights
        )

        return round(score, 4), breakdown, "\n".join(feedback_parts)

    @classmethod
    def grade(
        cls,
        classification_action: Optional[CustomerSupportAction],
        reply_action: Optional[CustomerSupportAction],
        escalation_action: Optional[CustomerSupportAction],
        ticket: CustomerTicket,
    ) -> Tuple[float, Dict[str, float], str]:
        feedback_parts = []

        # Classification
        if classification_action:
            cls_score, _, cls_feedback = EasyGrader.grade(
                classification_action, ticket
            )
            feedback_parts.append("=== Classification ===")
            feedback_parts.append(cls_feedback)
        else:
            cls_score = 0.0
            feedback_parts.append("=== Classification ===\n❌ Not provided")

        # Reply
        reply_text = reply_action.draft_reply if reply_action else None
        reply_score, _, reply_feedback = MediumGrader.grade_reply(
            reply_text, ticket
        )
        feedback_parts.append("\n=== Reply Quality ===")
        feedback_parts.append(reply_feedback)

        # Escalation
        esc_score, _, esc_feedback = cls.grade_escalation(
            escalation_action, ticket
        )
        feedback_parts.append("\n=== Escalation Decision ===")
        feedback_parts.append(esc_feedback)

        total = (
            cls_score * cls.CLASSIFICATION_WEIGHT
            + reply_score * cls.REPLY_WEIGHT
            + esc_score * cls.ESCALATION_WEIGHT
        )
        # OpenEnv strict requirement: (0, 1) exclusive
        total = max(1e-6, min(1 - 1e-6, total))

        return (
            round(total, 4),
            {
                "classification": max(1e-6, min(1 - 1e-6, float(cls_score))),
                "reply": max(1e-6, min(1 - 1e-6, float(reply_score))),
                "escalation": max(1e-6, min(1 - 1e-6, float(esc_score))),
            },
            "\n".join(feedback_parts),
        )

    @staticmethod
    def _infer_expected_level(ticket: CustomerTicket) -> EscalationLevel:
        """Infer the expected escalation level from ticket content."""
        body_lower = ticket.body.lower()
        subject_lower = ticket.subject.lower()
        combined = body_lower + " " + subject_lower

        if any(w in combined for w in ["legal", "attorney", "lawyer", "lawsuit", "gdpr"]):
            return EscalationLevel.LEGAL
        if any(w in combined for w in [
            "manager", "executive", "data breach", "deceased",
            "estate", "power of attorney",
        ]):
            return EscalationLevel.MANAGER
        return EscalationLevel.SUPERVISOR

    @staticmethod
    def _is_adjacent_level(
        level: Optional[EscalationLevel], expected: EscalationLevel
    ) -> bool:
        order = [
            EscalationLevel.NONE,
            EscalationLevel.SUPERVISOR,
            EscalationLevel.MANAGER,
            EscalationLevel.LEGAL,
        ]
        if level is None:
            return False
        return abs(order.index(level) - order.index(expected)) == 1

    @staticmethod
    def _get_escalation_keywords(ticket: CustomerTicket) -> list:
        """Return keywords that a good escalation reason should mention."""
        keywords = []
        body_lower = ticket.body.lower()
        if "legal" in body_lower or "attorney" in body_lower:
            keywords.extend(["legal", "attorney", "compliance"])
        if "breach" in body_lower:
            keywords.extend(["security", "breach", "data"])
        if "deceased" in body_lower or "estate" in body_lower:
            keywords.extend(["deceased", "sensitive", "estate"])
        if "gdpr" in body_lower:
            keywords.extend(["gdpr", "privacy", "regulation", "compliance"])
        if "angry" == (ticket.true_sentiment or ""):
            keywords.extend(["frustrated", "escalat", "chargeback"])
        if "production" in body_lower or "revenue" in body_lower:
            keywords.extend(["production", "revenue", "critical", "impact"])
        if "enterprise" in body_lower or "bulk" in body_lower:
            keywords.extend(["enterprise", "sales", "bulk"])
        if not keywords:
            keywords = ["escalat", "priority", "urgent"]
        return keywords
