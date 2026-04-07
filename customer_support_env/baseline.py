#!/usr/bin/env python3
"""
Baseline Inference Script — runs an OpenAI model against all 3 tasks.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

The script interacts with the environment LOCALLY (no HTTP server needed)
and produces reproducible scores on every task.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

from openai import OpenAI


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customer_support_env import (
    CustomerSupportAction,
    CustomerSupportEnv,
    TicketCategory,
    TicketPriority,
    TicketSentiment,
    EscalationLevel,
)


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    return OpenAI(api_key=api_key)


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")




def build_classification_prompt(ticket_text: str, task_desc: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Respond with ONLY valid JSON:
{{
  "classify_category": "<billing|technical|account|shipping|product|general>",
  "classify_priority": "<low|medium|high|urgent>",
  "classify_sentiment": "<positive|neutral|negative|angry>"
}}"""


def build_reply_prompt(ticket_text: str, task_desc: str, prev_feedback: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Previous step feedback:
{prev_feedback}

Respond with ONLY valid JSON:
{{
  "draft_reply": "<your professional reply to the customer>"
}}"""


def build_escalation_prompt(
    ticket_text: str, task_desc: str, prev_feedback: str
) -> str:
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


def call_llm(client: OpenAI, prompt: str) -> dict:
    """Call the LLM and parse JSON response."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert customer support agent. "
                    "Always respond with valid JSON only, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fence if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def parse_action(raw: dict) -> CustomerSupportAction:
    """Convert raw LLM JSON dict to a typed action."""
    kwargs = {}

    if "classify_category" in raw and raw["classify_category"]:
        try:
            kwargs["classify_category"] = TicketCategory(raw["classify_category"])
        except ValueError:
            pass

    if "classify_priority" in raw and raw["classify_priority"]:
        try:
            kwargs["classify_priority"] = TicketPriority(raw["classify_priority"])
        except ValueError:
            pass

    if "classify_sentiment" in raw and raw["classify_sentiment"]:
        try:
            kwargs["classify_sentiment"] = TicketSentiment(raw["classify_sentiment"])
        except ValueError:
            pass

    if "draft_reply" in raw and raw["draft_reply"]:
        kwargs["draft_reply"] = raw["draft_reply"]

    if "escalation_level" in raw and raw["escalation_level"]:
        try:
            kwargs["escalation_level"] = EscalationLevel(raw["escalation_level"])
        except ValueError:
            pass

    if "escalation_reason" in raw:
        kwargs["escalation_reason"] = raw["escalation_reason"]

    if "internal_notes" in raw:
        kwargs["internal_notes"] = raw["internal_notes"]

    return CustomerSupportAction(**kwargs)





def run_easy(client: OpenAI, env: CustomerSupportEnv, seed: int = 42) -> List[float]:
    """Run the easy task on all easy tickets."""
    from customer_support_env.tasks.easy import EasyTask

    scores = []
    for i, tid in enumerate(EasyTask.ticket_ids):
        obs = env.reset(task_id="easy_classify", ticket_id=tid, seed=seed + i)
        ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

        prompt = build_classification_prompt(ticket_text, obs.task_description)
        raw = call_llm(client, prompt)
        action = parse_action(raw)

        result = env.step(action)
        scores.append(result.observation.reward)
        print(
            f"  [Easy] {tid}: {result.observation.reward:.2f}  "
            f"| {result.observation.feedback}"
        )
    return scores


def run_medium(client: OpenAI, env: CustomerSupportEnv, seed: int = 42) -> List[float]:
    """Run the medium task on all medium tickets."""
    from customer_support_env.tasks.medium import MediumTask

    scores = []
    for i, tid in enumerate(MediumTask.ticket_ids):
        obs = env.reset(task_id="medium_reply", ticket_id=tid, seed=seed + i)
        ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

        
        prompt1 = build_classification_prompt(ticket_text, obs.task_description)
        raw1 = call_llm(client, prompt1)
        action1 = parse_action(raw1)
        result1 = env.step(action1)

        
        prompt2 = build_reply_prompt(
            ticket_text,
            result1.observation.task_description,
            result1.observation.feedback or "",
        )
        raw2 = call_llm(client, prompt2)
        action2 = parse_action(raw2)
        result2 = env.step(action2)

        scores.append(result2.observation.reward)
        print(f"  [Medium] {tid}: {result2.observation.reward:.2f}")

    return scores


def run_hard(client: OpenAI, env: CustomerSupportEnv, seed: int = 42) -> List[float]:
    """Run the hard task on all hard tickets."""
    from customer_support_env.tasks.hard import HardTask

    scores = []
    for i, tid in enumerate(HardTask.ticket_ids):
        obs = env.reset(task_id="hard_resolution", ticket_id=tid, seed=seed + i)
        ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

        
        prompt1 = build_classification_prompt(ticket_text, obs.task_description)
        raw1 = call_llm(client, prompt1)
        action1 = parse_action(raw1)
        result1 = env.step(action1)

        
        prompt2 = build_reply_prompt(
            ticket_text,
            result1.observation.task_description,
            result1.observation.feedback or "",
        )
        raw2 = call_llm(client, prompt2)
        action2 = parse_action(raw2)
        result2 = env.step(action2)

        
        prompt3 = build_escalation_prompt(
            ticket_text,
            result2.observation.task_description,
            result2.observation.feedback or "",
        )
        raw3 = call_llm(client, prompt3)
        action3 = parse_action(raw3)
        result3 = env.step(action3)

        scores.append(result3.observation.reward)
        print(f"  [Hard] {tid}: {result3.observation.reward:.2f}")

    return scores





def main():
    print("=" * 60)
    print("  Customer Support OpenEnv — Baseline Evaluation")
    print(f"  Model: {MODEL}")
    print("=" * 60)

    client = get_client()
    env = CustomerSupportEnv()

    results: Dict[str, Dict] = {}
    overall_scores = []

    
    print("\n📋 Task 1: Ticket Classification (Easy)")
    print("-" * 40)
    t0 = time.time()
    easy_scores = run_easy(client, env)
    easy_avg = sum(easy_scores) / len(easy_scores) if easy_scores else 0
    results["easy_classify"] = {
        "scores": easy_scores,
        "average": round(easy_avg, 4),
        "time_seconds": round(time.time() - t0, 1),
    }
    overall_scores.extend(easy_scores)
    print(f"  Average: {easy_avg:.4f}")

    
    print("\n📝 Task 2: Draft a Reply (Medium)")
    print("-" * 40)
    t0 = time.time()
    medium_scores = run_medium(client, env)
    medium_avg = sum(medium_scores) / len(medium_scores) if medium_scores else 0
    results["medium_reply"] = {
        "scores": medium_scores,
        "average": round(medium_avg, 4),
        "time_seconds": round(time.time() - t0, 1),
    }
    overall_scores.extend(medium_scores)
    print(f"  Average: {medium_avg:.4f}")

    
    print("\n🔥 Task 3: Full Resolution (Hard)")
    print("-" * 40)
    t0 = time.time()
    hard_scores = run_hard(client, env)
    hard_avg = sum(hard_scores) / len(hard_scores) if hard_scores else 0
    results["hard_resolution"] = {
        "scores": hard_scores,
        "average": round(hard_avg, 4),
        "time_seconds": round(time.time() - t0, 1),
    }
    overall_scores.extend(hard_scores)
    print(f"  Average: {hard_avg:.4f}")

    
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print("\n" + "=" * 60)
    print("  📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Easy  (classification):  {easy_avg:.4f}")
    print(f"  Medium (classify+reply): {medium_avg:.4f}")
    print(f"  Hard  (full resolution): {hard_avg:.4f}")
    print(f"  ---------------------------------")
    print(f"  Overall Average:         {overall_avg:.4f}")
    print("=" * 60)

    
    results["overall_average"] = round(overall_avg, 4)
    results["model"] = MODEL
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to outputs/baseline_results.json")


if __name__ == "__main__":
    main()
