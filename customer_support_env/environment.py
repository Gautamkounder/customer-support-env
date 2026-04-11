"""
Customer Support Resolution Environment.

Full OpenEnv-compatible environment implementing step() / reset() / state().
Supports three task difficulties: easy, medium, hard.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import (
    CustomerSupportAction,
    CustomerSupportObservation,
    CustomerSupportState,
    CustomerTicket,
    StepResult,
)
from .tasks import EasyTask, MediumTask, HardTask
from .graders import EasyGrader, MediumGrader, HardGrader


DATA_PATH = Path(__file__).parent / "data" / "tickets.json"


class CustomerSupportEnv:
    """
    OpenEnv-compatible Customer Support Resolution environment.

    API:
        reset(task_id, ticket_id=None, seed=None) → Observation
        step(action)                              → StepResult
        state()                                   → State
    """

    TASKS = {
        "easy_classify": EasyTask,
        "medium_reply": MediumTask,
        "hard_resolution": HardTask,
    }

    def __init__(self):
        self._tickets: Dict[str, CustomerTicket] = self._load_tickets()
        self._state: Optional[CustomerSupportState] = None
        self._current_ticket: Optional[CustomerTicket] = None
        self._task: Optional[Any] = None
        self._actions: List[CustomerSupportAction] = []
        self._step_rewards: List[float] = []
        self._step_feedbacks: List[str] = []

    # Data loading

    @staticmethod
    def _load_tickets() -> Dict[str, CustomerTicket]:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {t["ticket_id"]: CustomerTicket(**t) for t in raw}

    # reset()

    def reset(
        self,
        task_id: str = "easy_classify",
        ticket_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> CustomerSupportObservation:
        """
        Initialize a new episode.

        Args:
            task_id: One of easy_classify, medium_reply, hard_resolution
            ticket_id: Specific ticket to use (random if omitted)
            seed: Random seed for reproducibility
        """
        if task_id not in self.TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {list(self.TASKS.keys())}"
            )

        if seed is not None:
            random.seed(seed)

        task_cls = self.TASKS[task_id]
        self._task = task_cls

        # Pick a ticket
        if ticket_id:
            if ticket_id not in self._tickets:
                raise ValueError(f"Unknown ticket_id: {ticket_id}")
            self._current_ticket = self._tickets[ticket_id]
        else:
            available_ids = [
                tid for tid in task_cls.ticket_ids if tid in self._tickets
            ]
            chosen_id = random.choice(available_ids)
            self._current_ticket = self._tickets[chosen_id]

        # Initialize state
        self._state = CustomerSupportState(
            episode_id=str(uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=task_cls.max_steps,
            done=False,
            cumulative_reward=0.0,
            ticket_id=self._current_ticket.ticket_id,
        )
        self._actions = []
        self._step_rewards = []
        self._step_feedbacks = []

        # Build observation (strip ground truth)
        sanitized_ticket = task_cls.strip_ground_truth(self._current_ticket)

        # Get task description for step 0
        if hasattr(task_cls, "get_description"):
            description = task_cls.get_description(0)
        else:
            description = task_cls.description

        return CustomerSupportObservation(
            ticket=sanitized_ticket,
            task_description=description,
            step_count=0,
            max_steps=task_cls.max_steps,
            done=False,
            reward=0.0,
            feedback=None,
            metadata={
                "task_id": task_id,
                "task_name": task_cls.name,
                "difficulty": task_cls.difficulty,
                "episode_id": self._state.episode_id,
            },
        )

    # step()

    def step(self, action: CustomerSupportAction) -> StepResult:
        """
        Execute one step. Returns StepResult(observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._state.step_count += 1
        self._actions.append(action)

        task_id = self._state.task_id
        step_idx = self._state.step_count - 1  # 0-indexed
        is_last_step = self._state.step_count >= self._state.max_steps

        # Grade based on task type
        reward = 0.0
        feedback = ""
        info: Dict[str, Any] = {}

        if task_id == "easy_classify":
            reward, breakdown, feedback = EasyGrader.grade(
                action, self._current_ticket
            )
            info["breakdown"] = breakdown
            self._state.done = True

        elif task_id == "medium_reply":
            if step_idx == 0:
                # Step 1: classification — give partial reward
                r, breakdown, feedback = EasyGrader.grade(
                    action, self._current_ticket
                )
                reward = r * MediumGrader.CLASSIFICATION_WEIGHT
                info["step_type"] = "classification"
                info["breakdown"] = breakdown
            elif step_idx == 1:
                # Step 2: reply — grade full task
                r_full, breakdown, feedback = MediumGrader.grade(
                    self._actions[0],  # classification from step 1
                    action,            # reply from step 2
                    self._current_ticket,
                )
                # Subtract already-given classification reward
                already_given = sum(self._step_rewards)
                reward = max(0.0, r_full - already_given)
                info["step_type"] = "reply"
                info["breakdown"] = breakdown
                info["total_score"] = r_full
                self._state.done = True

            if is_last_step:
                self._state.done = True

        elif task_id == "hard_resolution":
            if step_idx == 0:
                r, breakdown, feedback = EasyGrader.grade(
                    action, self._current_ticket
                )
                reward = r * HardGrader.CLASSIFICATION_WEIGHT
                info["step_type"] = "classification"
                info["breakdown"] = breakdown
            elif step_idx == 1:
                # Reply step — give partial reply reward
                reply_score, _, reply_fb = MediumGrader.grade_reply(
                    action.draft_reply, self._current_ticket
                )
                reward = reply_score * HardGrader.REPLY_WEIGHT
                feedback = reply_fb
                info["step_type"] = "reply"
                info["reply_score"] = max(1e-6, min(1 - 1e-6, float(reply_score)))
            elif step_idx == 2:
                # Escalation + final grading
                r_full, breakdown, feedback = HardGrader.grade(
                    self._actions[0] if len(self._actions) > 0 else None,
                    self._actions[1] if len(self._actions) > 1 else None,
                    action,
                    self._current_ticket,
                )
                already_given = sum(self._step_rewards)
                reward = max(0.0, r_full - already_given)
                info["step_type"] = "escalation"
                info["breakdown"] = breakdown
                info["total_score"] = r_full
                self._state.done = True

            if is_last_step:
                self._state.done = True

        # Penalize clearly empty/no-op actions
        if self._is_empty_action(action):
            reward = reward - 0.1
            feedback += "\n⚠️ Penalty: empty action detected (-0.1)"

        # Enforce OpenEnv strictly (0, 1) bounds on the final cumulative score
        if self._state.done:
            projected_total = self._state.cumulative_reward + reward
            clamped_total = max(1e-6, min(1 - 1e-6, projected_total))
            # Adjust the current step's reward so the sum matches clamped_total
            reward = clamped_total - self._state.cumulative_reward
        else:
            # For intermediate steps, just ensure it doesn't drop below 0
            reward = max(0.0, reward)

        self._step_rewards.append(reward)
        self._step_feedbacks.append(feedback)
        self._state.cumulative_reward += reward

        # Build next observation
        next_description = ""
        if not self._state.done and hasattr(self._task, "get_description"):
            next_description = self._task.get_description(self._state.step_count)
        elif not self._state.done and hasattr(self._task, "description"):
            next_description = self._task.description

        sanitized = self._task.strip_ground_truth(self._current_ticket)

        obs = CustomerSupportObservation(
            ticket=sanitized,
            task_description=next_description,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            done=self._state.done,
            reward=self._state.cumulative_reward,
            feedback=feedback,
            metadata={
                "task_id": task_id,
                "episode_id": self._state.episode_id,
                **info,
            },
        )

        # Clamp step reward to strict (0, 1) after rounding
        step_reward = round(reward, 4)
        step_reward = max(1e-6, min(1 - 1e-6, step_reward))

        return StepResult(
            observation=obs,
            reward=step_reward,
            done=self._state.done,
            info=info,
        )

    # state()

    def state(self) -> CustomerSupportState:
        """Return the current internal state."""
        if self._state is None:
            return CustomerSupportState(episode_id="none", task_id="none")
        return self._state

    # Helpers

    @staticmethod
    def _is_empty_action(action: CustomerSupportAction) -> bool:
        """Check if the agent sent a completely empty action."""
        return (
            action.classify_category is None
            and action.classify_priority is None
            and action.classify_sentiment is None
            and (action.draft_reply is None or action.draft_reply.strip() == "")
            and action.escalation_level is None
        )
