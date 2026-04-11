#!/usr/bin/env python3
"""
Local validation test — verifies ALL scores from ALL tasks
are strictly in (0, 1) i.e. > 0.0 and < 1.0.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from customer_support_env import (
    CustomerSupportEnv,
    CustomerSupportAction,
    TicketCategory,
    TicketPriority,
    TicketSentiment,
    EscalationLevel,
)
from customer_support_env.tasks.easy import EasyTask
from customer_support_env.tasks.medium import MediumTask
from customer_support_env.tasks.hard import HardTask

env = CustomerSupportEnv()
all_scores = []
errors = []

def check_score(label, score):
    """Check score is strictly in (0, 1)"""
    if score <= 0.0 or score >= 1.0:
        errors.append(f"❌ FAIL {label}: score={score} is NOT in (0, 1)")
        return False
    else:
        print(f"  ✅ {label}: {score:.6f}")
        return True

# =========================================================
# TEST 1: Easy Task — Classification only
# =========================================================
print("=" * 60)
print("TEST 1: Easy Task (Classification)")
print("=" * 60)

# Test with perfect action
for tid in EasyTask.ticket_ids:
    obs = env.reset(task_id="easy_classify", ticket_id=tid, seed=42)
    ticket = env._current_ticket
    action = CustomerSupportAction(
        classify_category=ticket.true_category,
        classify_priority=ticket.true_priority,
        classify_sentiment=ticket.true_sentiment,
    )
    result = env.step(action)
    check_score(f"easy/{tid}/perfect/reward", result.reward)
    check_score(f"easy/{tid}/perfect/cumulative", result.observation.reward)
    all_scores.append(result.observation.reward)

# Test with completely wrong action
for tid in EasyTask.ticket_ids[:2]:
    obs = env.reset(task_id="easy_classify", ticket_id=tid, seed=42)
    action = CustomerSupportAction()  # empty = worst case
    result = env.step(action)
    check_score(f"easy/{tid}/empty/cumulative", result.observation.reward)
    all_scores.append(result.observation.reward)

# =========================================================
# TEST 2: Medium Task — Classify + Reply
# =========================================================
print("\n" + "=" * 60)
print("TEST 2: Medium Task (Classify + Reply)")
print("=" * 60)

for tid in MediumTask.ticket_ids:
    obs = env.reset(task_id="medium_reply", ticket_id=tid, seed=42)
    ticket = env._current_ticket

    # Step 1: perfect classification
    action1 = CustomerSupportAction(
        classify_category=ticket.true_category,
        classify_priority=ticket.true_priority,
        classify_sentiment=ticket.true_sentiment,
    )
    result1 = env.step(action1)
    check_score(f"medium/{tid}/step1/cumulative", result1.observation.reward)

    # Step 2: good reply
    action2 = CustomerSupportAction(
        draft_reply=f"Dear {ticket.customer_name.split()[0]}, thank you for reaching out. "
                    f"I understand your concern regarding {ticket.subject}. "
                    f"We sincerely apologize for the inconvenience and will resolve this promptly. "
                    f"Please allow us 24-48 hours. Best regards."
    )
    result2 = env.step(action2)
    check_score(f"medium/{tid}/step2/reward", result2.reward)
    check_score(f"medium/{tid}/step2/cumulative", result2.observation.reward)
    all_scores.append(result2.observation.reward)

# Test with empty actions
for tid in MediumTask.ticket_ids[:1]:
    obs = env.reset(task_id="medium_reply", ticket_id=tid, seed=42)
    action1 = CustomerSupportAction()
    result1 = env.step(action1)
    action2 = CustomerSupportAction()
    result2 = env.step(action2)
    check_score(f"medium/{tid}/empty/cumulative", result2.observation.reward)
    all_scores.append(result2.observation.reward)

# =========================================================
# TEST 3: Hard Task — Classify + Reply + Escalate
# =========================================================
print("\n" + "=" * 60)
print("TEST 3: Hard Task (Classify + Reply + Escalate)")
print("=" * 60)

for tid in HardTask.ticket_ids:
    obs = env.reset(task_id="hard_resolution", ticket_id=tid, seed=42)
    ticket = env._current_ticket

    # Step 1: perfect classification
    action1 = CustomerSupportAction(
        classify_category=ticket.true_category,
        classify_priority=ticket.true_priority,
        classify_sentiment=ticket.true_sentiment,
    )
    result1 = env.step(action1)
    check_score(f"hard/{tid}/step1/cumulative", result1.observation.reward)

    # Step 2: good reply
    action2 = CustomerSupportAction(
        draft_reply=f"Dear {ticket.customer_name.split()[0]}, thank you for contacting us. "
                    f"I understand your frustration with {ticket.subject}. "
                    f"We take this very seriously and will escalate to the appropriate team. "
                    f"Your satisfaction is our priority. We will follow up within 24 hours."
    )
    result2 = env.step(action2)
    check_score(f"hard/{tid}/step2/cumulative", result2.observation.reward)

    # Step 3: escalation
    action3 = CustomerSupportAction(
        escalation_level=EscalationLevel.SUPERVISOR if ticket.requires_escalation else EscalationLevel.NONE,
        escalation_reason="This ticket requires immediate attention due to its complexity and customer impact. Escalating for proper resolution.",
        internal_notes="Priority case - requires supervisor review.",
    )
    result3 = env.step(action3)
    check_score(f"hard/{tid}/step3/reward", result3.reward)
    check_score(f"hard/{tid}/step3/cumulative", result3.observation.reward)
    all_scores.append(result3.observation.reward)

# Test with empty actions
for tid in HardTask.ticket_ids[:1]:
    obs = env.reset(task_id="hard_resolution", ticket_id=tid, seed=42)
    for step in range(3):
        action = CustomerSupportAction()
        result = env.step(action)
    check_score(f"hard/{tid}/empty/cumulative", result.observation.reward)
    all_scores.append(result.observation.reward)

# =========================================================
# FINAL REPORT
# =========================================================
print("\n" + "=" * 60)
print("FINAL VALIDATION REPORT")
print("=" * 60)

if errors:
    print(f"\n❌ {len(errors)} ERRORS FOUND:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print(f"\n✅ ALL {len(all_scores)} scores validated — strictly in (0, 1)")
    print(f"   Min score: {min(all_scores):.10f}")
    print(f"   Max score: {max(all_scores):.10f}")
    print(f"   All > 0.0: {all(s > 0.0 for s in all_scores)}")
    print(f"   All < 1.0: {all(s < 1.0 for s in all_scores)}")
    print(f"\n🎉 READY FOR SUBMISSION!")
    sys.exit(0)
