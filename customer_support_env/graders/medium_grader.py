"""
Medium Grader — scores classify + reply task.
All scores strictly in open interval (0, 1) — never 0.0 or 1.0.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..models import CustomerSupportAction, CustomerTicket
from .easy_grader import EasyGrader, _clamp, _MIN, _MAX


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
            # Return _MIN not 0.0 — strict (0,1) requirement
            return _MIN, {"reply_provided": _MIN}, "X No reply provided"

        reply_lower = reply.lower()

        # -- Has reply --
        breakdown["reply_provided"] = _MAX
        feedback_parts.append("OK Reply provided")

        # -- Minimum length --
        if len(reply.strip()) >= 50:
            breakdown["min_length"] = _MAX
            feedback_parts.append(f"OK Reply length adequate ({len(reply.strip())} chars)")
        else:
            breakdown["min_length"] = _MIN
            feedback_parts.append(f"X Reply too short ({len(reply.strip())} chars, need 50+)")

        # -- Customer name --
        first_name = ticket.customer_name.split()[0].lower()
        if first_name in reply_lower:
            breakdown["uses_name"] = _MAX
            feedback_parts.append("OK Addresses customer by name")
        else:
            breakdown["uses_name"] = _MIN
            feedback_parts.append("X Does not address customer by name")

        # -- Professional tone --
        caps_ratio = sum(1 for c in reply if c.isupper()) / max(len(reply), 1)
        rude_patterns = [
            r"\byour fault\b", r"\byou should have\b", r"\bnot my problem\b",
            r"\bfigure it out\b", r"\bdeal with it\b",
        ]
        is_rude = any(re.search(p, reply_lower) for p in rude_patterns)
        if caps_ratio < 0.4 and not is_rude:
            breakdown["professional_tone"] = _MAX
            feedback_parts.append("OK Professional tone")
        else:
            breakdown["professional_tone"] = _MIN
            reason = "excessive caps" if caps_ratio >= 0.4 else "unprofessional language"
            feedback_parts.append(f"X Tone issue: {reason}")

        # -- References specifics --
        specifics_found = 0
        specifics_total = 0
        specific_patterns = re.findall(
            r'(?:order|#|TKT|TRK|ACC|ENT|req_|error)\s*[-#]?\s*\w+',
            ticket.body,
            re.IGNORECASE,
        )
        if specific_patterns:
            specifics_total = min(len(specific_patterns), 3)
            for pattern in specific_patterns[:3]:
                key = re.findall(r'[A-Z]*[-#]?\d+\w*', pattern)
                if key and any(k.lower() in reply_lower for k in key):
                    specifics_found += 1
        if specifics_total > 0:
            raw_ratio = specifics_found / specifics_total
            breakdown["references_specifics"] = _clamp(raw_ratio)
            feedback_parts.append(
                f"{'OK' if specifics_found == specifics_total else '~~'} "
                f"References specifics: {specifics_found}/{specifics_total}"
            )
        else:
            breakdown["references_specifics"] = _MAX
            feedback_parts.append("OK No specific references needed")

        # -- Resolution points coverage --
        if ticket.expected_resolution_points:
            points_hit = cls._check_resolution_points(
                reply, ticket.expected_resolution_points
            )
            total_points = len(ticket.expected_resolution_points)
            raw_ratio = points_hit / total_points
            breakdown["resolution_points"] = _clamp(raw_ratio)
            feedback_parts.append(
                f"{'OK' if points_hit == total_points else '~~'} "
                f"Resolution points: {points_hit}/{total_points}"
            )
        else:
            breakdown["resolution_points"] = _MAX

        # Weighted score
        weights = {
            "reply_provided":       0.10 / cls.REPLY_WEIGHT,
            "min_length":           0.10 / cls.REPLY_WEIGHT,
            "uses_name":            0.05 / cls.REPLY_WEIGHT,
            "professional_tone":    0.10 / cls.REPLY_WEIGHT,
            "references_specifics": 0.15 / cls.REPLY_WEIGHT,
            "resolution_points":    0.20 / cls.REPLY_WEIGHT,
        }
        w_total = sum(weights.values())
        score = sum(breakdown[k] * weights[k] / w_total for k in weights if k in breakdown)

        return round(_clamp(score), 4), {k: _clamp(v) for k, v in breakdown.items()}, "\n".join(feedback_parts)

    @classmethod
    def grade(
        cls,
        classification_action: Optional[CustomerSupportAction],
        reply_action: Optional[CustomerSupportAction],
        ticket: CustomerTicket,
    ) -> Tuple[float, Dict[str, float], str]:
        """Grade the full medium task. All scores strictly in (0, 1)."""
        feedback_parts = []

        # Classification
        if classification_action:
            cls_score, cls_breakdown, cls_feedback = cls.grade_classification(
                classification_action, ticket
            )
            feedback_parts.append("=== Classification ===")
            feedback_parts.append(cls_feedback)
        else:
            cls_score = _MIN
            feedback_parts.append("=== Classification ===\nX No classification provided")

        # Reply
        reply_text = reply_action.draft_reply if reply_action else None
        reply_score, reply_breakdown, reply_feedback = cls.grade_reply(
            reply_text, ticket
        )
        feedback_parts.append("\n=== Reply Quality ===")
        feedback_parts.append(reply_feedback)

        total = cls_score * cls.CLASSIFICATION_WEIGHT + reply_score * cls.REPLY_WEIGHT
        # OpenEnv strict requirement: (0, 1) exclusive
        total = _clamp(total)
        breakdown = {
            "classification": _clamp(cls_score),
            "reply":          _clamp(reply_score),
        }
        return round(total, 4), breakdown, "\n".join(feedback_parts)

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
            matched = sum(1 for w in key_words if w in reply_words)
            if matched / len(key_words) >= 0.4:
                hits += 1
        return hits
