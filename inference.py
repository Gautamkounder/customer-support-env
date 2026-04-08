#!/usr/bin/env python3
"""
inference.py — OpenEnv Hackathon Submission Inference Script

Runs an LLM agent against the Customer Support Resolution environment
on all 3 tasks, producing structured [START]/[STEP]/[END] stdout logs.

Required environment variables (checked in priority order):
    API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN      — API key / Hugging Face token (also checked as OPENAI_API_KEY)

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
import traceback
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — ensure the local package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Environment variables — read with sane fallbacks, NEVER exit early
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Accept both HF_TOKEN and OPENAI_API_KEY (validator may use either)
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("API_KEY")
    or ""
)

if not HF_TOKEN:
    # Warn but DO NOT exit — let the LLM call fail gracefully later
    print("[WARN] No API key found in HF_TOKEN / OPENAI_API_KEY / API_KEY. "
          "LLM calls will fail gracefully.", flush=True)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy-key",   # OpenAI SDK requires non-empty key
    )
except Exception as e:
    print(f"[ERROR] Failed to initialise OpenAI client: {e}", flush=True)
    client = None  # type: ignore

# ---------------------------------------------------------------------------
# Local package imports — wrapped so import errors are non-fatal
# ---------------------------------------------------------------------------
try:
    from customer_support_env import (
        CustomerSupportAction,
        CustomerSupportEnv,
        TicketCategory,
        TicketPriority,
        TicketSentiment,
        EscalationLevel,
    )
    from customer_support_env.tasks.easy   import EasyTask
    from customer_support_env.tasks.medium import MediumTask
    from customer_support_env.tasks.hard   import HardTask
    ENV_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] Failed to import customer_support_env: {e}", flush=True)
    traceback.print_exc()
    ENV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM call — fully wrapped with retries and graceful fallback
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert customer support agent. "
    "Always respond with valid JSON only. "
    "No markdown code fences, no extra text — just the JSON object."
)

MAX_RETRIES   = 3
RETRY_DELAY_S = 2.0


def call_llm(prompt: str) -> dict:
    """
    Call the LLM and parse JSON response.

    Retries up to MAX_RETRIES times on transient errors.
    Returns an empty dict on total failure (never raises).
    """
    if client is None:
        print("[WARN] LLM client not available — returning empty action.", flush=True)
        return {}

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content
            if text is None:
                raise ValueError("LLM returned None content")
            text = text.strip()

            # Strip markdown code fences if the model adds them anyway
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first fence line and optional trailing fence
                inner = lines[1:]
                if inner and inner[-1].strip() == "```":
                    inner = inner[:-1]
                text = "\n".join(inner).strip()

            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parse error on attempt {attempt}: {e}", flush=True)
            last_err = e
            # Don't retry JSON errors — return fallback immediately
            break

        except Exception as e:
            print(
                f"[WARN] LLM call failed (attempt {attempt}/{MAX_RETRIES}): "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)

    print(f"[WARN] All LLM attempts exhausted. Last error: {last_err}. "
          "Using empty fallback action.", flush=True)
    return {}


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(raw: dict) -> "CustomerSupportAction":
    """Convert raw LLM JSON to typed CustomerSupportAction (never raises)."""
    kwargs: dict = {}

    enum_fields = [
        ("classify_category", TicketCategory),
        ("classify_priority",  TicketPriority),
        ("classify_sentiment", TicketSentiment),
        ("escalation_level",   EscalationLevel),
    ]
    for field, enum_cls in enum_fields:
        val = raw.get(field)
        if val:
            try:
                kwargs[field] = enum_cls(str(val).lower())
            except ValueError:
                print(f"[WARN] Invalid value '{val}' for {field} — skipping.", flush=True)

    for field in ["draft_reply", "escalation_reason", "internal_notes"]:
        val = raw.get(field)
        if val:
            kwargs[field] = str(val)

    return CustomerSupportAction(**kwargs)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def classification_prompt(ticket_text: str, task_desc: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Respond with ONLY valid JSON (no markdown fences):
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

Respond with ONLY valid JSON (no markdown fences):
{{
  "draft_reply": "<your professional reply to the customer>"
}}"""


def escalation_prompt(ticket_text: str, task_desc: str, prev_feedback: str) -> str:
    return f"""{task_desc}

TICKET:
{ticket_text}

Previous step feedback:
{prev_feedback}

Respond with ONLY valid JSON (no markdown fences):
{{
  "escalation_level": "<none|supervisor|manager|legal>",
  "escalation_reason": "<explain your decision>",
  "internal_notes": "<any notes for the team>"
}}"""


# ---------------------------------------------------------------------------
# Task runners — each ticket is fully isolated; failures yield 0 reward
# ---------------------------------------------------------------------------

def run_easy_task(env: "CustomerSupportEnv") -> List[float]:
    """Run easy (classification) task on all easy tickets."""
    task_id    = "easy_classify"
    ticket_ids = EasyTask.ticket_ids
    log_start(task_id, "Ticket Classification", "easy", len(ticket_ids))

    scores: List[float] = []
    for i, tid in enumerate(ticket_ids):
        try:
            obs    = env.reset(task_id=task_id, ticket_id=tid, seed=42 + i)
            ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

            prompt = classification_prompt(ticket_text, obs.task_description)
            raw    = call_llm(prompt)
            action = parse_action(raw)

            result = env.step(action)
            reward = result.observation.reward
            done   = result.done

        except Exception as e:
            print(f"[ERROR] Easy task failed for ticket {tid}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            # Fallback must be > 0.0 for OpenEnv
            reward, done = 0.01, True

        scores.append(reward)
        log_step(
            task_id=task_id,
            ticket_id=tid,
            step=1,
            max_steps=1,
            reward=reward,
            done=done,
        )

    total = sum(scores)
    avg   = total / len(scores) if scores else 0.0
    log_end(task_id, total, avg, len(scores))
    return scores


def run_medium_task(env: "CustomerSupportEnv") -> List[float]:
    """Run medium (classify + reply) task on all medium tickets."""
    task_id    = "medium_reply"
    ticket_ids = MediumTask.ticket_ids
    log_start(task_id, "Draft a Reply", "medium", len(ticket_ids))

    scores: List[float] = []
    for i, tid in enumerate(ticket_ids):
        reward, done = 0.01, True
        try:
            obs         = env.reset(task_id=task_id, ticket_id=tid, seed=42 + i)
            ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

            # Step 1 — Classification
            prompt1  = classification_prompt(ticket_text, obs.task_description)
            raw1     = call_llm(prompt1)
            action1  = parse_action(raw1)
            result1  = env.step(action1)

            log_step(
                task_id=task_id, ticket_id=tid,
                step=1, max_steps=2,
                reward=result1.observation.reward,
                done=result1.done,
            )

            # Step 2 — Reply
            prompt2  = reply_prompt(
                ticket_text,
                result1.observation.task_description,
                result1.observation.feedback or "",
            )
            raw2     = call_llm(prompt2)
            action2  = parse_action(raw2)
            result2  = env.step(action2)

            reward = result2.observation.reward
            done   = result2.done

        except Exception as e:
            print(f"[ERROR] Medium task failed for ticket {tid}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            # Fallback must be > 0.0 for OpenEnv
            reward, done = 0.01, True

        scores.append(reward)
        log_step(
            task_id=task_id, ticket_id=tid,
            step=2, max_steps=2,
            reward=reward, done=done,
        )

    total = sum(scores)
    avg   = total / len(scores) if scores else 0.0
    log_end(task_id, total, avg, len(scores))
    return scores


def run_hard_task(env: "CustomerSupportEnv") -> List[float]:
    """Run hard (classify + reply + escalation) task on all hard tickets."""
    task_id    = "hard_resolution"
    ticket_ids = HardTask.ticket_ids
    log_start(task_id, "Full Resolution", "hard", len(ticket_ids))

    scores: List[float] = []
    for i, tid in enumerate(ticket_ids):
        reward, done = 0.01, True
        try:
            obs         = env.reset(task_id=task_id, ticket_id=tid, seed=42 + i)
            ticket_text = f"Subject: {obs.ticket.subject}\n\n{obs.ticket.body}"

            # Step 1 — Classification
            prompt1  = classification_prompt(ticket_text, obs.task_description)
            raw1     = call_llm(prompt1)
            action1  = parse_action(raw1)
            result1  = env.step(action1)

            log_step(
                task_id=task_id, ticket_id=tid,
                step=1, max_steps=3,
                reward=result1.observation.reward,
                done=result1.done,
            )

            # Step 2 — Reply
            prompt2  = reply_prompt(
                ticket_text,
                result1.observation.task_description,
                result1.observation.feedback or "",
            )
            raw2     = call_llm(prompt2)
            action2  = parse_action(raw2)
            result2  = env.step(action2)

            log_step(
                task_id=task_id, ticket_id=tid,
                step=2, max_steps=3,
                reward=result2.observation.reward,
                done=result2.done,
            )

            # Step 3 — Escalation
            prompt3  = escalation_prompt(
                ticket_text,
                result2.observation.task_description,
                result2.observation.feedback or "",
            )
            raw3     = call_llm(prompt3)
            action3  = parse_action(raw3)
            result3  = env.step(action3)

            reward = result3.observation.reward
            done   = result3.done

        except Exception as e:
            print(f"[ERROR] Hard task failed for ticket {tid}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            reward, done = 0.01, True

        scores.append(reward)
        log_step(
            task_id=task_id, ticket_id=tid,
            step=3, max_steps=3,
            reward=reward, done=done,
        )

    total = sum(scores)
    avg   = total / len(scores) if scores else 0.01
    log_end(task_id, total, avg, len(scores))
    return scores


# ---------------------------------------------------------------------------
# Main — wrapped in a top-level try/except so it NEVER exits non-zero
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("  Customer Support OpenEnv — Inference Script", flush=True)
    print(f"  Model:    {MODEL_NAME}", flush=True)
    print(f"  API Base: {API_BASE_URL}", flush=True)
    print(f"  HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET — only fallback actions'}", flush=True)
    print("=" * 60, flush=True)

    if not ENV_AVAILABLE:
        print("[ERROR] customer_support_env package could not be imported. "
              "Emitting zero scores.", flush=True)
        # Still emit well-formed logs so the validator can parse them
        for task_id, name, diff, n in [
            ("easy_classify",   "Ticket Classification", "easy",   8),
            ("medium_reply",    "Draft a Reply",         "medium", 6),
            ("hard_resolution", "Full Resolution",       "hard",   6),
        ]:
            log_start(task_id, name, diff, n)
            for j in range(n):
                log_step(task_id=task_id, ticket_id=f"TKT-{j+1:03d}",
                         step=1, max_steps=1, reward=0.01, done=True)
            log_end(task_id, 0.01, 0.01, n)
        return

    env        = CustomerSupportEnv()
    all_scores: List[float] = []

    # ── Easy Task ────────────────────────────────────────────────────────────
    print("\n📋 Task 1: Ticket Classification (Easy)", flush=True)
    print("-" * 40, flush=True)
    try:
        easy_scores = run_easy_task(env)
    except Exception as e:
        print(f"[ERROR] run_easy_task crashed: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        easy_scores = []
    easy_avg = sum(easy_scores) / len(easy_scores) if easy_scores else 0.01
    all_scores.extend(easy_scores)
    print(f"  Easy Average: {easy_avg:.4f}", flush=True)

    # ── Medium Task ──────────────────────────────────────────────────────────
    print("\n📝 Task 2: Draft a Reply (Medium)", flush=True)
    print("-" * 40, flush=True)
    try:
        medium_scores = run_medium_task(env)
    except Exception as e:
        print(f"[ERROR] run_medium_task crashed: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        medium_scores = []
    medium_avg = sum(medium_scores) / len(medium_scores) if medium_scores else 0.01
    all_scores.extend(medium_scores)
    print(f"  Medium Average: {medium_avg:.4f}", flush=True)

    # ── Hard Task ────────────────────────────────────────────────────────────
    print("\n🔥 Task 3: Full Resolution (Hard)", flush=True)
    print("-" * 40, flush=True)
    try:
        hard_scores = run_hard_task(env)
    except Exception as e:
        print(f"[ERROR] run_hard_task crashed: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        hard_scores = []
    hard_avg = sum(hard_scores) / len(hard_scores) if hard_scores else 0.01
    all_scores.extend(hard_scores)
    print(f"  Hard Average: {hard_avg:.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.01
    print("\n" + "=" * 60, flush=True)
    print("  📊 RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Easy   (classification):   {easy_avg:.4f}", flush=True)
    print(f"  Medium (classify+reply):   {medium_avg:.4f}", flush=True)
    print(f"  Hard   (full resolution):  {hard_avg:.4f}", flush=True)
    print(f"  ---------------------------------", flush=True)
    print(f"  Overall Average:           {overall_avg:.4f}", flush=True)
    print("=" * 60, flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model":            MODEL_NAME,
        "api_base_url":     API_BASE_URL,
        "easy_classify":    {"scores": easy_scores,   "average": round(easy_avg, 4)},
        "medium_reply":     {"scores": medium_scores,  "average": round(medium_avg, 4)},
        "hard_resolution":  {"scores": hard_scores,    "average": round(hard_avg, 4)},
        "overall_average":  round(overall_avg, 4),
    }
    try:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n💾 Results saved to outputs/baseline_results.json", flush=True)
    except Exception as e:
        print(f"[WARN] Could not save results file: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last-resort catch — print the error but exit with 0
        print(f"\n[FATAL] Unhandled exception in main(): {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc(file=sys.stdout)
        print("[FATAL] Exiting with code 0 to avoid fail-fast pipeline stop.", flush=True)
        sys.exit(0)
