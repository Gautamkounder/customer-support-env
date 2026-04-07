"""
Customer Support Resolution Environment — Pydantic Models.

Defines the typed Action, Observation, and State models used across
the environment's step() / reset() / state() API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Enums

class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SHIPPING = "shipping"
    PRODUCT = "product"
    GENERAL = "general"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class EscalationLevel(str, Enum):
    NONE = "none"
    SUPERVISOR = "supervisor"
    MANAGER = "manager"
    LEGAL = "legal"


# Ticket

class CustomerTicket(BaseModel):
    """A single customer support ticket."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_name: str = Field(..., description="Customer display name")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Full message body")
    true_category: Optional[TicketCategory] = Field(
        None, description="Ground-truth category (hidden from agent)"
    )
    true_priority: Optional[TicketPriority] = Field(
        None, description="Ground-truth priority (hidden from agent)"
    )
    true_sentiment: Optional[TicketSentiment] = Field(
        None, description="Ground-truth sentiment (hidden from agent)"
    )
    requires_escalation: Optional[bool] = Field(
        None, description="Whether the ticket truly needs escalation"
    )
    expected_resolution_points: Optional[List[str]] = Field(
        None, description="Key points a good reply must address"
    )


# Action

class CustomerSupportAction(BaseModel):
    """
    Agent action.  At minimum the agent must supply a `classify_category`.
    Medium / hard tasks also require `draft_reply` and escalation fields.
    """
    classify_category: Optional[TicketCategory] = Field(
        None, description="Agent's classification of the ticket"
    )
    classify_priority: Optional[TicketPriority] = Field(
        None, description="Agent's priority assessment"
    )
    classify_sentiment: Optional[TicketSentiment] = Field(
        None, description="Agent's sentiment assessment"
    )
    draft_reply: Optional[str] = Field(
        None, description="Agent's drafted reply to the customer"
    )
    escalation_level: Optional[EscalationLevel] = Field(
        None, description="Agent's escalation decision"
    )
    escalation_reason: Optional[str] = Field(
        None, description="Why the agent chose to escalate"
    )
    internal_notes: Optional[str] = Field(
        None, description="Internal notes attached to the ticket"
    )


# Observation

class CustomerSupportObservation(BaseModel):
    """What the agent sees after reset() or step()."""
    ticket: Optional[CustomerTicket] = Field(
        None, description="Current ticket (ground-truth fields stripped)"
    )
    task_description: str = Field(
        "", description="Human-readable task instructions"
    )
    step_count: int = Field(0, description="Steps taken so far")
    max_steps: int = Field(1, description="Maximum steps allowed")
    done: bool = Field(False, description="Whether the episode is finished")
    reward: float = Field(0.0, description="Cumulative reward so far")
    feedback: Optional[str] = Field(
        None, description="Grader feedback from the last action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extra info dict"
    )


# State

class CustomerSupportState(BaseModel):
    """Internal episode state exposed via state()."""
    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field("", description="Which task is active")
    step_count: int = Field(0, description="Steps taken this episode")
    max_steps: int = Field(1, description="Max steps for this task")
    done: bool = Field(False, description="Episode finished?")
    cumulative_reward: float = Field(0.0, description="Total reward so far")
    ticket_id: Optional[str] = Field(None, description="Current ticket id")


# StepResult

class StepResult(BaseModel):
    """Canonical return type of step()."""
    observation: CustomerSupportObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
