# gradient_text_tool.py
import os
from typing import Optional, Dict, Any

import httpx
from pydantic import BaseModel, Field


class GradientChatInput(BaseModel):
    prompt: str = Field(..., description="User prompt to send to Gradient text model")
    system: Optional[str] = Field(
        default="You are a concise, helpful assistant.",
        description="Optional system instruction for the model.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model override (otherwise uses GRADIENT_TEXT_MODEL).",
    )
    max_tokens: int = Field(default=600, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0, le=2.0)


class GradientConfigError(RuntimeError):
    pass


class GradientRequestError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


async def gradient_chat(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    base_url = os.getenv("GRADIENT_BASE_URL", "https://inference.do-ai.run/v1").rstrip("/")
    api_key = os.getenv("GRADIENT_API_KEY")  # (your model access key)
    default_model = os.getenv("GRADIENT_TEXT_MODEL", "llama3-8b-instruct")
    chosen_model = model or default_model

    if not api_key:
        raise GradientConfigError("Missing GRADIENT_API_KEY")

    sys_msg = (system or "You are a concise, helpful assistant.").strip()
    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": chosen_model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    url = f"{base_url}/chat/completions"

    timeout = httpx.Timeout(60.0, connect=10.0)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        try:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.RequestError as e:
            raise GradientRequestError(f"Network error calling Gradient: {e}") from e

    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise GradientRequestError(
            f"Gradient returned {resp.status_code}",
            status_code=resp.status_code,
            body=body,
        )

    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return (text or "").strip() or "(empty response)"
