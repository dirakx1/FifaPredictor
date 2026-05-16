"""
Base class for all football agents.
Mirrors MiroFish's OASIS agent model:
  - Each agent has a rich persona (profile) loaded at init
  - Each agent maintains a message history across rounds (= long-term memory)
  - Each agent acts asynchronously
  - Agents see the shared MatchState broadcast before acting
"""
import asyncio
import json
import os
from typing import Any, Optional

import anthropic

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Shared async client (one per process)
_async_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _async_client


class FootballAgent:
    """
    An LLM-backed agent with persistent round-to-round memory.

    The conversation history IS the memory — exactly as MiroFish agents retain
    context across simulation rounds via their message list.
    """

    ROLE = "football_agent"   # override in subclasses

    def __init__(self, agent_id: str, persona: str, system_prompt: str):
        self.agent_id = agent_id
        self.persona = persona
        self.system_prompt = system_prompt
        # Message history persists across all rounds — this is the agent's memory
        self._history: list[dict] = []

    def _inject_state(self, state_broadcast: str) -> None:
        """Append the current match state as a new user message (the 'feed')."""
        self._history.append({
            "role": "user",
            "content": state_broadcast,
        })

    def _record_response(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})

    async def _call_llm(self, extra_instruction: str = "") -> str:
        messages = list(self._history)
        if extra_instruction:
            messages.append({"role": "user", "content": extra_instruction})

        resp = await _get_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=self.system_prompt,
            messages=messages,
        )
        return resp.content[0].text.strip()

    async def act(self, state_broadcast: str, extra_instruction: str = "") -> dict[str, Any]:
        """
        Called once per round. Returns a structured action dict.
        Subclasses override `_parse_action` to interpret the raw LLM text.
        """
        self._inject_state(state_broadcast)
        raw = await self._call_llm(extra_instruction)
        self._record_response(raw)
        return self._parse_action(raw)

    def _parse_action(self, raw: str) -> dict[str, Any]:
        """Parse LLM output into a structured action. Subclasses override this."""
        # Strip markdown fences if present
        text = raw
        if "```" in raw:
            parts = raw.split("```")
            for p in parts:
                stripped = p.strip().lstrip("json").strip()
                if stripped.startswith("{"):
                    text = stripped
                    break
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": raw}

    def reset(self) -> None:
        """Clear memory for a new simulation run."""
        self._history = []
