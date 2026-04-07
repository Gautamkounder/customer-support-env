#!/usr/bin/env python3
"""
inference.py — OpenEnv Hackathon Submission Inference Script

Runs an LLM agent against the Customer Support Resolution environment
on all 3 tasks, producing structured [START]/[STEP]/[END] stdout logs.

Required environment variables:
    API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN      — API key / Hugging Face token

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=sk-...
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

from openai import OpenAI


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from customer_support_env import (
    CustomerSupportAction,
    CustomerSupportEnv,
    TicketCategory,
    TicketPriority,
    TicketSentiment,
    EscalationLevel,
)
from customer_support_env.tasks.easy import EasyTask
from customer_support_env.tasks.medium import MediumTask
from customer_support_env.tasks.hard import HardTask



API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required", flush=True)
    sys.exit(1)


client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)




def log_start(task_id: str, task_name: str, difficulty: str, num_tickets: int):
    """Emit [START] log."""
    print(
        f"[START] task_id={task_id} task_name={task_name} "
        f"difficulty={difficulty} num_tickets={num_tickets} "
        f"model={MODEL_NAME}",
        flush=True,
    )


def log_step(
    task_id: str,
    ticket_id: str,
    step: int,
    max_steps: int,
    reward: float,
    done: bool,
    **extra,
):
    """Emit [STEP] log."""
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    print(
        f"[STEP] task_id={task_id} ticket_id={ticket_id} "
        f"step={step} max_steps={max_steps} "
        f"reward={reward:.4f} done={done}"
        + (f" {extra_str}" if extra_str else ""),
        flush=True,
    )


def log_end(task_id: str, total_reward: float, avg_reward: float, num_episodes: int):
    """Emit [END] log."""
    print(
        f"[END] task_id={task_id} total_reward={total_reward:.4f} "
        f"avg_reward={avg_reward:.4f} num_episodes={num_episodes}",
        flush=True,
    )




SYSTEM_PROMPT = (
    "You are an expert customer support agent. "
    "Always respond with valid JSON only. "
    "No markdown code fences, no extra text — just the JSON object."
)


def call_llm(prompt: str) -> dict:
    """Call the LLM and parse JSON response."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])  # Skip first line
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def parse_action(raw: dict) -> CustomerSupportAction:
    """Convert raw LLM JSON to typed CustomerSupportAction."""
    kwargs = {}

    for field, enum_cls in [
        ("classify_category", TicketCategory),
        ("classify_priority", TicketPriority),
        ("classify_sentiment", TicketSentiment),
        ("escalation_level", EscalationLevel),
    ]:
        if field in raw and raw[field]:
            try:
                kwargs[field] = enum_cls(raw[field])
            except ValueError:
                pass

    for field in ["draft_reply", "escalation_reason", "internal_notes"]:
        if field in raw and raw[field]:
            kwargs[field] = str(raw[field])

    return CustomerSupportAction(**kwargs)





def classification_prompt(ticket_text: str, task_desc: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Respond with ONLY valid JSON:
{{
  "classify_category": "<billing|technical|account|shipping|product|general>",
  "classify_priority": "<low|medium|high|urgent>",
  "classify_sentiment": "<positive|neutral|negative|angry>"
}}"""


def reply_prompt(ticket_text: str, task_desc: str, prev_feedback: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Previous step feedback:
{prev_feedback}

Respond with ONLY valid JSON:
{{
  "draft_reply": "<your professional reply to the customer>"
}}"""


def escalation_prompt(ticket_text: str, task_desc: str, prev_feedback: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Previous step feedback:
{prev_feedback}

Respond with ONLY valid JSON:
{{
  "escalation_level": "<none|supervisor|manager|legal>",
  "escalation_reason": "<explain your decision>",
  "internal_notes": "<any notes for the team>"
}}"""





def run_easy_task(env: CustomerSupportEnv) -> List[float]:
    """Run easy task (classification) on all easy tickets."""
    task_id = "easy_classify"
    ticket_ids = EasyTask.ticket_ids
    log_start(task_id, "Ticket Classification", "easy", len(ticket_ids))

    scores = []
    for i, tid in enumerate(ticket_ids):
        obs = env.reset(task_id=task_id, ticket_id=tid, seed=42 + i)
        ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

        prompt = classification_prompt(ticket_text, obs.task_description)
        raw = call_llm(prompt)
        action = parse_action(raw)

        result = env.step(action)
        scores.append(result.observation.reward)

        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=1,
            max_steps=1,
            reward=result.observation.reward,
            done=result.done,
        )

    total = sum(scores)
    avg = total / len(scores) if scores else 0
    log_end(task_id, total, avg, len(scores))
    return scores


def run_medium_task(env: CustomerSupportEnv) -> List[float]:
    """Run medium task (classify + reply) on all medium tickets."""
    task_id = "medium_reply"
    ticket_ids = MediumTask.ticket_ids
    log_start(task_id, "Draft a Reply", "medium", len(ticket_ids))

    scores = []
    for i, tid in enumerate(ticket_ids):
        obs = env.reset(task_id=task_id, ticket_id=tid, seed=42 + i)
        ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

        # Step 1: Classification
        prompt1 = classification_prompt(ticket_text, obs.task_description)
        raw1 = call_llm(prompt1)
        action1 = parse_action(raw1)
        result1 = env.step(action1)

        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=1,
            max_steps=2,
            reward=result1.observation.reward,
            done=result1.done,
        )

        # Step 2: Reply
        prompt2 = reply_prompt(
            ticket_text,
            result1.observation.task_description,
            result1.observation.feedback or "",
        )
        raw2 = call_llm(prompt2)
        action2 = parse_action(raw2)
        result2 = env.step(action2)

        scores.append(result2.observation.reward)

        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=2,
            max_steps=2,
            reward=result2.observation.reward,
            done=result2.done,
        )

    total = sum(scores)
    avg = total / len(scores) if scores else 0
    log_end(task_id, total, avg, len(scores))
    return scores


def run_hard_task(env: CustomerSupportEnv) -> List[float]:
    """Run hard task (classify + reply + escalation) on all hard tickets."""
    task_id = "hard_resolution"
    ticket_ids = HardTask.ticket_ids
    log_start(task_id, "Full Resolution", "hard", len(ticket_ids))

    scores = []
    for i, tid in enumerate(ticket_ids):
        obs = env.reset(task_id=task_id, ticket_id=tid, seed=42 + i)
        ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

        # Step 1: Classification
        prompt1 = classification_prompt(ticket_text, obs.task_description)
        raw1 = call_llm(prompt1)
        action1 = parse_action(raw1)
        result1 = env.step(action1)

        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=1,
            max_steps=3,
            reward=result1.observation.reward,
            done=result1.done,
        )

        # Step 2: Reply
        prompt2 = reply_prompt(
            ticket_text,
            result1.observation.task_description,
            result1.observation.feedback or "",
        )
        raw2 = call_llm(prompt2)
        action2 = parse_action(raw2)
        result2 = env.step(action2)

        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=2,
            max_steps=3,
            reward=result2.observation.reward,
            done=result2.done,
        )

        # Step 3: Escalation
        prompt3 = escalation_prompt(
            ticket_text,
            result2.observation.task_description,
            result2.observation.feedback or "",
        )
        raw3 = call_llm(prompt3)
        action3 = parse_action(raw3)
        result3 = env.step(action3)

        scores.append(result3.observation.reward)

        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=3,
            max_steps=3,
            reward=result3.observation.reward,
            done=result3.done,
        )

    total = sum(scores)
    avg = total / len(scores) if scores else 0
    log_end(task_id, total, avg, len(scores))
    return scores





def main():
    print("=" * 60, flush=True)
    print("  Customer Support OpenEnv — Inference Script", flush=True)
    print(f"  Model: {MODEL_NAME}", flush=True)
    print(f"  API Base: {API_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    env = CustomerSupportEnv()
    all_scores = []

    # Easy Task
    print("\n📋 Task 1: Ticket Classification (Easy)", flush=True)
    print("-" * 40, flush=True)
    easy_scores = run_easy_task(env)
    easy_avg = sum(easy_scores) / len(easy_scores) if easy_scores else 0
    all_scores.extend(easy_scores)
    print(f"  Easy Average: {easy_avg:.4f}", flush=True)

    # Medium Task
    print("\n📝 Task 2: Draft a Reply (Medium)", flush=True)
    print("-" * 40, flush=True)
    medium_scores = run_medium_task(env)
    medium_avg = sum(medium_scores) / len(medium_scores) if medium_scores else 0
    all_scores.extend(medium_scores)
    print(f"  Medium Average: {medium_avg:.4f}", flush=True)

    # Hard Task
    print("\n🔥 Task 3: Full Resolution (Hard)", flush=True)
    print("-" * 40, flush=True)
    hard_scores = run_hard_task(env)
    hard_avg = sum(hard_scores) / len(hard_scores) if hard_scores else 0
    all_scores.extend(hard_scores)
    print(f"  Hard Average: {hard_avg:.4f}", flush=True)

    # Summary
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print("\n" + "=" * 60, flush=True)
    print("  📊 RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Easy  (classification):   {easy_avg:.4f}", flush=True)
    print(f"  Medium (classify+reply):  {medium_avg:.4f}", flush=True)
    print(f"  Hard  (full resolution):  {hard_avg:.4f}", flush=True)
    print(f"  ---------------------------------", flush=True)
    print(f"  Overall Average:          {overall_avg:.4f}", flush=True)
    print("=" * 60, flush=True)

    # Save results
    results = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "easy_classify": {"scores": easy_scores, "average": round(easy_avg, 4)},
        "medium_reply": {"scores": medium_scores, "average": round(medium_avg, 4)},
        "hard_resolution": {"scores": hard_scores, "average": round(hard_avg, 4)},
        "overall_average": round(overall_avg, 4),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to outputs/baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
