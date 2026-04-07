# 🎧 Customer Support Resolution Environment

> **An OpenEnv-compatible environment for training AI agents on real-world customer support ticket resolution.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 What is this?

This environment simulates the work of a **customer support agent** — a task that every company on earth needs. An AI agent interacts with realistic support tickets and must:

1. **Classify** tickets by category, priority, and sentiment
2. **Draft** professional, empathetic replies
3. **Escalate** complex issues to the right team

The environment provides **deterministic scoring** (0.0–1.0) with **partial progress rewards**, making it ideal for RL post-training and agent evaluation.

---

## 🏗️ Architecture

```
project-root/
├── inference.py               # Inference script (uses API_BASE_URL, MODEL_NAME, HF_TOKEN)
├── Dockerfile                 # Containerized deployment for HF Spaces
└── customer_support_env/
    ├── __init__.py            # Package exports
    ├── models.py              # Pydantic: Action, Observation, State
    ├── environment.py         # Core step()/reset()/state() logic
    ├── server.py              # FastAPI HTTP server
    ├── openenv.yaml           # OpenEnv manifest
    ├── requirements.txt       # Python dependencies
    ├── tasks/
    │   ├── easy.py            # Task 1: Classification
    │   ├── medium.py          # Task 2: Classify + Reply
    │   └── hard.py            # Task 3: Full Resolution
    ├── graders/
    │   ├── easy_grader.py     # Deterministic classification scorer
    │   ├── medium_grader.py   # Classification + reply quality scorer
    │   └── hard_grader.py     # Full resolution scorer
    └── data/
        └── tickets.json       # 15 realistic support tickets
```

---

## 📋 Tasks

### Task 1: Ticket Classification (Easy)
- **Steps:** 1
- **Objective:** Classify a ticket by category, priority, and sentiment
- **Scoring:** Exact match with partial credit for adjacent answers
  - Category match: 40%
  - Priority match: 30% (half credit if off by one level)
  - Sentiment match: 30% (half credit for adjacent sentiments)

### Task 2: Draft a Reply (Medium)
- **Steps:** 2
- **Objective:** Classify the ticket, then draft a professional reply
- **Scoring:**
  - Classification accuracy: 30%
  - Reply quality: 70%
    - Reply provided + adequate length
    - Addresses customer by name
    - Professional tone (no ALL CAPS, no rudeness)
    - References ticket-specific details (order #, error codes)
    - Covers expected resolution points

### Task 3: Full Resolution (Hard)
- **Steps:** 3
- **Objective:** Classify, reply, AND make escalation decisions
- **Scoring:**
  - Classification: 20%
  - Reply quality: 45%
  - Escalation decision: 35%
    - Correct escalation yes/no
    - Appropriate escalation level
    - Quality of escalation reasoning

---

## 🎯 Action Space

```python
class CustomerSupportAction(BaseModel):
    classify_category: Optional[TicketCategory]    # billing|technical|account|shipping|product|general
    classify_priority: Optional[TicketPriority]    # low|medium|high|urgent
    classify_sentiment: Optional[TicketSentiment]  # positive|neutral|negative|angry
    draft_reply: Optional[str]                     # Professional reply to customer
    escalation_level: Optional[EscalationLevel]    # none|supervisor|manager|legal
    escalation_reason: Optional[str]               # Why escalate
    internal_notes: Optional[str]                  # Team notes
```

## 👁️ Observation Space

```python
class CustomerSupportObservation(BaseModel):
    ticket: Optional[CustomerTicket]  # Subject, body, customer name
    task_description: str             # Instructions for current step
    step_count: int                   # Current step (0-indexed)
    max_steps: int                    # Total steps for this task
    done: bool                        # Episode finished?
    reward: float                     # Cumulative reward
    feedback: Optional[str]          # Grader feedback from last action
    metadata: Dict[str, Any]         # Extra info
```

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone and install
cd customer_support_env
pip install -r requirements.txt

# Run the server
python -m customer_support_env.server
# → Server at http://localhost:8000
# → API docs at http://localhost:8000/docs
```

### Python API (Direct)

```python
from customer_support_env import CustomerSupportEnv, CustomerSupportAction, TicketCategory, TicketPriority, TicketSentiment

env = CustomerSupportEnv()

# Reset with easy task
obs = env.reset(task_id="easy_classify", seed=42)
print(obs.ticket.subject)  # "Can't access my account after password reset"
print(obs.task_description)  # Classification instructions

# Take an action
action = CustomerSupportAction(
    classify_category=TicketCategory.ACCOUNT,
    classify_priority=TicketPriority.HIGH,
    classify_sentiment=TicketSentiment.NEUTRAL,
)
result = env.step(action)
print(f"Score: {result.reward}")  # 1.0 for perfect classification
print(result.observation.feedback)  # Detailed grader feedback
```

### HTTP API

```bash
# Health check
curl http://localhost:8000/health

# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_classify", "seed": 42}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"classify_category": "account", "classify_priority": "high", "classify_sentiment": "neutral"}}'

# State
curl http://localhost:8000/state
```

### Docker

```bash
docker build -t customer-support-env .
docker run -p 8000:8000 customer-support-env
```

---

## 📊 Baseline Scores

Run the inference script (uses OpenAI-compatible API):

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

The script emits structured `[START]`, `[STEP]`, and `[END]` logs for automated evaluation.

### Expected Baseline Scores (gpt-4o-mini)

| Task | Difficulty | Avg Score | Description |
|------|-----------|-----------|-------------|
| Ticket Classification | Easy | ~0.85 | Category + Priority + Sentiment |
| Draft a Reply | Medium | ~0.72 | Classification + Reply Quality |
| Full Resolution | Hard | ~0.65 | Classify + Reply + Escalation |
| **Overall** | — | **~0.74** | Weighted average |

> Scores are reproducible with `seed=42` and `temperature=0.0`.

---

## 🏗️ Deploying to Hugging Face Spaces

1. Create a new Space on huggingface.co (Docker type)
2. Push this directory:

```bash
# Install OpenEnv CLI
pip install openenv-core

# Push to HF Spaces
openenv push --repo-id YOUR_USERNAME/customer-support-env

# Or manually with git
cd customer_support_env
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/customer-support-env
git add .
git commit -m "Initial customer support env"
git push origin main
```

3. Tag the Space with `openenv`

---

## 🎓 Ticket Dataset

The environment includes **15 diverse, realistic tickets** covering:

| Category | Count | Examples |
|----------|-------|---------|
| Billing | 3 | Double charges, refund demands, legal estate issues |
| Technical | 3 | API outages, app crashes, Salesforce sync failures |
| Account | 3 | Password resets, data breach concerns, GDPR requests |
| Shipping | 2 | Wrong items, stuck shipments |
| Product | 1 | Feature requests |
| General | 3 | Plan inquiries, bulk orders, positive feedback |

Sentiments range from enthusiastic praise to legal threats. Priorities from informational to production-critical. Some tickets require escalation to legal or management.

---

## 🔧 Reward Function Design

The reward function provides **meaningful signal throughout the trajectory**:

1. **Partial progress:** Each step in a multi-step task earns proportional reward
2. **Partial credit:** Adjacent classifications (e.g., medium vs. high priority) earn half credit
3. **Quality rubric:** Reply grading checks multiple dimensions (tone, specifics, resolution points)
4. **Penalties:** Empty or no-op actions receive a -0.1 penalty
5. **Deterministic:** All grading is programmatic with no LLM-in-the-loop

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
