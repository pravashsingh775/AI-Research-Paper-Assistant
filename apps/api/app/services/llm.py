from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def clean_json_response(raw_text: str) -> str:
    """Extract valid JSON from raw LLM output, stripping markdown fences if present."""
    text = raw_text.strip()
    # Strip markdown code blocks ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    # Find outer JSON boundaries [ ... ] or { ... }
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = text.rfind("}")
        if last_brace != -1:
            text = text[first_brace : last_brace + 1]
    elif first_bracket != -1:
        last_bracket = text.rfind("]")
        if last_bracket != -1:
            text = text[first_bracket : last_bracket + 1]
    return text.strip()


class LLMService:
    """Production-grade multi-provider asynchronous LLM client."""

    @staticmethod
    def resolve_key(provider: str = "gemini", custom_key: str | None = None) -> str | None:
        """Resolve API key from user header/custom key or environment variables."""
        if custom_key and custom_key.strip():
            return custom_key.strip()
        provider = provider.lower()
        if provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        elif provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    @classmethod
    async def generate_text(
        cls,
        prompt: str,
        system_instruction: str = "You are an expert scientific researcher and academic paper analyst.",
        provider: str = "gemini",
        model: str | None = None,
        custom_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str | None:
        """Generate text response using Gemini, OpenAI, or Anthropic."""
        key = cls.resolve_key(provider, custom_key)
        if not key:
            # If default provider has no key, check other available providers
            if os.getenv("GEMINI_API_KEY"):
                provider, key = "gemini", os.getenv("GEMINI_API_KEY")
            elif os.getenv("OPENAI_API_KEY"):
                provider, key = "openai", os.getenv("OPENAI_API_KEY")
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider, key = "anthropic", os.getenv("ANTHROPIC_API_KEY")
            else:
                return None

        provider = provider.lower()
        if provider == "gemini":
            model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return await cls._call_gemini(
                prompt, system_instruction, key, model_name, temperature, max_tokens
            )
        elif provider == "openai":
            model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return await cls._call_openai(
                prompt, system_instruction, key, model_name, temperature, max_tokens
            )
        elif provider == "anthropic":
            model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
            return await cls._call_anthropic(
                prompt, system_instruction, key, model_name, temperature, max_tokens
            )
        return None

    @classmethod
    async def generate_json(
        cls,
        prompt: str,
        schema_hint: str,
        system_instruction: str = "You are an expert academic research assistant. Return strictly valid JSON.",
        provider: str = "gemini",
        model: str | None = None,
        custom_key: str | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any] | list[Any] | None:
        """Generate and parse structured JSON output from LLM."""
        full_prompt = (
            f"{prompt}\n\n"
            f"STRICT REQUIREMENT: Output must be a single valid JSON object or array adhering to:\n"
            f"{schema_hint}\n"
            f"Do not include explanation, greetings, or preamble outside the JSON."
        )
        raw_text = await cls.generate_text(
            full_prompt,
            system_instruction=system_instruction,
            provider=provider,
            model=model,
            custom_key=custom_key,
            temperature=temperature,
            max_tokens=2500,
        )
        if not raw_text:
            return None

        cleaned = clean_json_response(raw_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(f"Initial JSON parse failed: {exc}. Attempting repair...")
            try:
                repaired = re.sub(r",\s*([\]}])", r"\1", cleaned)
                return json.loads(repaired)
            except Exception:
                logger.error(f"Failed to parse structured JSON from model: {raw_text[:200]}")
                return None

    @staticmethod
    async def _call_gemini(
        prompt: str,
        system_instruction: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": f"System Context:\n{system_instruction}\n\nTask:\n{prompt}"}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=35) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip()
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        if attempt < 2:
                            await asyncio.sleep(2**attempt)
                            continue
                        logger.warning(
                            f"Gemini API rate limit or error {resp.status_code}: {resp.text[:150]}"
                        )
                    else:
                        logger.warning(
                            f"Gemini API returned error status {resp.status_code}: {resp.text[:150]}"
                        )
                        return None
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.warning(f"Gemini API connection error: {exc}")
                return None
        return None

    @staticmethod
    async def _call_openai(
        prompt: str,
        system_instruction: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=35) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        if attempt < 2:
                            await asyncio.sleep(2**attempt)
                            continue
                    else:
                        return None
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.warning(f"OpenAI API connection error: {exc}")
                return None
        return None

    @staticmethod
    async def _call_anthropic(
        prompt: str,
        system_instruction: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_instruction,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=35) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["content"][0]["text"].strip()
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        if attempt < 2:
                            await asyncio.sleep(2**attempt)
                            continue
                    else:
                        return None
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.warning(f"Anthropic API error: {exc}")
                return None
        return None

    @classmethod
    async def verify_key(
        cls, provider: str, api_key: str, model: str | None = None
    ) -> dict[str, Any]:
        """Test and verify an API key provided by the user."""
        test_prompt = "Say 'OK' if you receive this message."
        res = await cls.generate_text(
            prompt=test_prompt,
            system_instruction="Reply with a short confirmation.",
            provider=provider,
            model=model,
            custom_key=api_key,
            max_tokens=20,
        )
        if res:
            return {
                "valid": True,
                "provider": provider,
                "model": model or "default",
                "message": "Connection successfully verified!",
            }
        return {
            "valid": False,
            "provider": provider,
            "model": model or "default",
            "message": f"Could not connect to {provider.title()} with the provided API key. Check key validity and permissions.",
        }
