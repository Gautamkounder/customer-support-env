"""
FastAPI server for the Customer Support OpenEnv environment.

Endpoints:
  GET  /health           → health check
  POST /reset            → reset episode
  POST /step             → take an action
  GET  /state            → get current state
  GET  /docs             → auto-generated API docs
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .environment import CustomerSupportEnv
from .models import (
    CustomerSupportAction,
    CustomerSupportObservation,
    CustomerSupportState,
    StepResult,
)


app = FastAPI(
    title="Customer Support Resolution Environment",
    description=(
        "An OpenEnv-compatible environment for training AI agents "
        "on customer support ticket resolution."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


env = CustomerSupportEnv()





class ResetRequest(BaseModel):
    task_id: str = "easy_classify"
    ticket_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: CustomerSupportAction


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "customer_support_env"
    version: str = "1.0.0"





@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/reset", response_model=CustomerSupportObservation)
async def reset(req: ResetRequest):
    try:
        obs = env.reset(
            task_id=req.task_id,
            ticket_id=req.ticket_id,
            seed=req.seed,
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest):
    try:
        result = env.step(req.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=CustomerSupportState)
async def state():
    return env.state()





def main():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "customer_support_env.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
