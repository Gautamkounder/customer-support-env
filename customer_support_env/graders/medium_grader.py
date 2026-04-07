"""
Medium Grader — scores classification + reply quality.

Overall scoring:
  • Classification accuracy:  0.30  (same formula as EasyGrader)
  • Reply quality:            0.70
    - Has reply at all:       0.10
    - Minimum length (50+ chars): 0.10
    - Addresses customer by name: 0.05
    - Professional tone (no ALL CAPS, no rudeness indicators): 0.10
    - References ticket specifics (order #, error, etc.): 0.15
    - Covers expected resolution points: 0.20
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..models import CustomerSupportAction, CustomerTicket
from .easy_grader import EasyGrader


class MediumGrader:
    """Grader for medium (classify + reply) task."""

    CLASSIFICATION_WEIGHT = 0.30
    REPLY_WEIGHT = 0.70

    @classmethod
    def grade_classification(
        cls, action: CustomerSupportAction, ticket: CustomerTicket
    ) -> Tuple[float, Dict[str, float], str]:
        return EasyGrader.grade(action, ticket)

    @classmethod
    def grade_reply(
        cls,
        reply: Optional[str],
        ticket: CustomerTicket,
    ) -> Tuple[float, Dict[str, float], str]:
        breakdown: Dict[str, float] = {}
        feedback_parts = []

        if not reply or not reply.strip():
            return 0.0, {"reply_provided": 0.0}, "❌ No reply provided"

        reply_lower = reply.lower()

        # ── Has reply ──
        breakdown["reply_provided"] = 1.0
        feedback_parts.append("✅ Reply provided")

        # ── Minimum length ──
        if len(reply.strip()) >= 50:
            breakdown["min_length"] = 1.0
            feedback_parts.append(f"✅ Reply length adequate ({len(reply.strip())} chars)")
        else:
            breakdown["min_length"] = 0.0
            feedback_parts.append(f"❌ Reply too short ({len(reply.strip())} chars, need 50+)")

        # ── Customer name ──
        first_name = ticket.customer_name.split()[0].lower()
        if first_name in reply_lower:
            breakdown["uses_name"] = 1.0
            feedback_parts.append("✅ Addresses customer by name")
        else:
            breakdown["uses_name"] = 0.0
            feedback_parts.append("❌ Does not address customer by name")

        # ── Professional tone ──
        caps_ratio = sum(1 for c in reply if c.isupper()) / max(len(reply), 1)
        rude_patterns = [
            r"\byour fault\b", r"\byou should have\b", r"\bnot my problem\b",
            r"\bfigure it out\b", r"\bdeal with it\b",
        ]
        is_rude = any(re.search(p, reply_lower) for p in rude_patterns)
        if caps_ratio < 0.4 and not is_rude:
            breakdown["professional_tone"] = 1.0
            feedback_parts.append("✅ Professional tone")
        else:
            breakdown["professional_tone"] = 0.0
            reason = "excessive caps" if caps_ratio >= 0.4 else "unprofessional language"
            feedback_parts.append(f"❌ Tone issue: {reason}")

        # ── References specifics ──
        specifics_found = 0
        specifics_total = 0
        # Check for order numbers, tracking numbers, error codes, account IDs
        specific_patterns = re.findall(
            r'(?:order|#|TKT|TRK|ACC|ENT|req_|error)\s*[-#]?\s*\w+',
            ticket.body,
            re.IGNORECASE,
        )
        if specific_patterns:
            specifics_total = min(len(specific_patterns), 3)
            for pattern in specific_patterns[:3]:
                # Extract the key identifier
                key = re.findall(r'[A-Z]*[-#]?\d+\w*', pattern)
                if key and any(k.lower() in reply_lower for k in key):
                    specifics_found += 1
        if specifics_total > 0:
            breakdown["references_specifics"] = specifics_found / specifics_total
            feedback_parts.append(
                f"{'✅' if specifics_found == specifics_total else '⚠️'} "
                f"References specifics: {specifics_found}/{specifics_total}"
            )
        else:
            breakdown["references_specifics"] = 1.0  # No specifics to check
            feedback_parts.append("✅ No specific references needed")

        # ── Resolution points coverage ──
        if ticket.expected_resolution_points:
            points_hit = cls._check_resolution_points(
                reply, ticket.expected_resolution_points
            )
            total_points = len(ticket.expected_resolution_points)
            breakdown["resolution_points"] = points_hit / total_points
            feedback_parts.append(
                f"{'✅' if points_hit == total_points else '⚠️'} "
                f"Resolution points: {points_hit}/{total_points}"
            )
        else:
            breakdown["resolution_points"] = 1.0

        # Weighted score
        weights = {
            "reply_provided": 0.10 / cls.REPLY_WEIGHT,
            "min_length": 0.10 / cls.REPLY_WEIGHT,
            "uses_name": 0.05 / cls.REPLY_WEIGHT,
            "professional_tone": 0.10 / cls.REPLY_WEIGHT,
            "references_specifics": 0.15 / cls.REPLY_WEIGHT,
            "resolution_points": 0.20 / cls.REPLY_WEIGHT,
        }
        # Normalize weights to sum to 1
        w_total = sum(weights.values())
        score = sum(breakdown[k] * weights[k] / w_total for k in weights if k in breakdown)

        return round(score, 4), breakdown, "\n".join(feedback_parts)

    @classmethod
    def grade(
        cls,
        classification_action: Optional[CustomerSupportAction],
        reply_action: Optional[CustomerSupportAction],
        ticket: CustomerTicket,
    ) -> Tuple[float, Dict[str, float], str]:
        """Grade the full medium task."""
        feedback_parts = []

        # Classification
        if classification_action:
            cls_score, cls_breakdown, cls_feedback = cls.grade_classification(
                classification_action, ticket
            )
            feedback_parts.append("=== Classification ===")
            feedback_parts.append(cls_feedback)
        else:
            cls_score = 0.0
            feedback_parts.append("=== Classification ===\n❌ No classification provided")

        # Reply
        reply_text = reply_action.draft_reply if reply_action else None
        reply_score, reply_breakdown, reply_feedback = cls.grade_reply(
            reply_text, ticket
        )
        feedback_parts.append("\n=== Reply Quality ===")
        feedback_parts.append(reply_feedback)

        total = cls_score * cls.CLASSIFICATION_WEIGHT + reply_score * cls.REPLY_WEIGHT
        return round(total, 4), {"classification": cls_score, "reply": reply_score}, "\n".join(feedback_parts)

    @staticmethod
    def _check_resolution_points(reply: str, points: List[str]) -> int:
        """
        Heuristic check: for each expected resolution point,
        look for semantic keyword overlap in the reply.
        """
        reply_lower = reply.lower()
        reply_words = set(re.findall(r'\w+', reply_lower))
        hits = 0
        for point in points:
            point_words = set(re.findall(r'\w+', point.lower()))
            # Remove very common words
            stop_words = {
                "the", "a", "an", "and", "or", "to", "for", "of", "in",
                "is", "it", "that", "this", "with", "be", "on", "at",
                "by", "from", "as", "any", "all", "if", "not", "but",
                "will", "can", "has", "have", "had", "do", "does",
            }
            key_words = point_words - stop_words
            if not key_words:
                hits += 1
                continue
            # Need at least 40% of key words present
            matched = sum(1 for w in key_words if w in reply_words)
            if matched / len(key_words) >= 0.4:
                hits += 1
        return hits
