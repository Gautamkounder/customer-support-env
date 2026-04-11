"""
Hard Grader — scores classification + reply + escalation decision.
All scores strictly in open interval (0, 1) — never 0.0 or 1.0.

Overall scoring:
  - Classification accuracy:   0.20
  - Reply quality:             0.45
  - Escalation decision:       0.35
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
from .easy_grader import EasyGrader, _clamp, _MIN, _MAX
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
            # Return _MIN not 0.0 — strict (0,1) requirement
            return _MIN, {"escalation_decision": _MIN}, "X No escalation action provided"

        requires = ticket.requires_escalation or False
        agent_escalates = (
            action.escalation_level is not None
            and action.escalation_level != EscalationLevel.NONE
        )

        # -- Correct decision --
        if requires == agent_escalates:
            breakdown["correct_decision"] = _MAX
            feedback_parts.append(
                f"OK Escalation decision correct "
                f"({'escalated' if agent_escalates else 'not escalated'})"
            )
        else:
            breakdown["correct_decision"] = _MIN
            if requires and not agent_escalates:
                feedback_parts.append("X Should have escalated this ticket but didn't")
            else:
                feedback_parts.append("X Escalated unnecessarily")

        # -- Appropriate level --
        if agent_escalates and requires:
            expected_level = cls._infer_expected_level(ticket)
            if action.escalation_level == expected_level:
                breakdown["appropriate_level"] = _MAX
                feedback_parts.append(f"OK Escalation level correct ({expected_level.value})")
            elif cls._is_adjacent_level(action.escalation_level, expected_level):
                breakdown["appropriate_level"] = 0.5
                feedback_parts.append(
                    f"~~ Escalation level close "
                    f"({action.escalation_level.value}, expected {expected_level.value})"
                )
            else:
                breakdown["appropriate_level"] = _MIN
                feedback_parts.append(
                    f"X Wrong escalation level "
                    f"({action.escalation_level.value}, expected {expected_level.value})"
                )
        elif not requires and not agent_escalates:
            breakdown["appropriate_level"] = _MAX  # Correctly didn't escalate
        else:
            breakdown["appropriate_level"] = _MIN

        # -- Reason quality --
        if action.escalation_reason and len(action.escalation_reason.strip()) >= 20:
            reason_lower = action.escalation_reason.lower()
            relevant_keywords = cls._get_escalation_keywords(ticket)
            keyword_hits = sum(1 for kw in relevant_keywords if kw in reason_lower)
            keyword_ratio = keyword_hits / max(len(relevant_keywords), 1)
            breakdown["reason_quality"] = _clamp(0.5 + keyword_ratio * 0.5)
            if keyword_ratio >= 0.3:
                feedback_parts.append(
                    f"OK Escalation reason is well-justified ({keyword_hits} relevant points)"
                )
            else:
                feedback_parts.append("~~ Escalation reason could be more specific")
        elif action.escalation_reason:
            breakdown["reason_quality"] = 0.3
            feedback_parts.append("~~ Escalation reason too brief")
        else:
            # requires=True  -> bad (no reason given): _MIN
            # requires=False -> ok (no reason needed): 0.8
            breakdown["reason_quality"] = _MIN if requires else 0.8
            if requires:
                feedback_parts.append("X No escalation reason provided")
            else:
                feedback_parts.append("OK No escalation reason needed")

        weights = {
            "correct_decision":  0.15 / cls.ESCALATION_WEIGHT,
            "appropriate_level": 0.10 / cls.ESCALATION_WEIGHT,
            "reason_quality":    0.10 / cls.ESCALATION_WEIGHT,
        }
        w_total = sum(weights.values())
        score = sum(
            breakdown.get(k, _MIN) * weights[k] / w_total
            for k in weights
        )

        return round(_clamp(score), 4), {k: _clamp(v) for k, v in breakdown.items()}, "\n".join(feedback_parts)

    @classmethod
    def grade(
        cls,
        classification_action: Optional[CustomerSupportAction],
        reply_action: Optional[CustomerSupportAction],
        escalation_action: Optional[CustomerSupportAction],
        ticket: CustomerTicket,
    ) -> Tuple[float, Dict[str, float], str]:
        """Grade the full hard task. All scores strictly in (0, 1)."""
        feedback_parts = []

        # Classification
        if classification_action:
            cls_score, _, cls_feedback = EasyGrader.grade(classification_action, ticket)
            feedback_parts.append("=== Classification ===")
            feedback_parts.append(cls_feedback)
        else:
            cls_score = _MIN
            feedback_parts.append("=== Classification ===\nX Not provided")

        # Reply
        reply_text = reply_action.draft_reply if reply_action else None
        reply_score, _, reply_feedback = MediumGrader.grade_reply(reply_text, ticket)
        feedback_parts.append("\n=== Reply Quality ===")
        feedback_parts.append(reply_feedback)

        # Escalation
        esc_score, _, esc_feedback = cls.grade_escalation(escalation_action, ticket)
        feedback_parts.append("\n=== Escalation Decision ===")
        feedback_parts.append(esc_feedback)

        total = (
            cls_score * cls.CLASSIFICATION_WEIGHT
            + reply_score * cls.REPLY_WEIGHT
            + esc_score * cls.ESCALATION_WEIGHT
        )
        # OpenEnv strict requirement: (0, 1) exclusive
        total = _clamp(total)

        return (
            float(total),
            {
                "classification": _clamp(cls_score),
                "reply":          _clamp(reply_score),
                "escalation":     _clamp(esc_score),
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
