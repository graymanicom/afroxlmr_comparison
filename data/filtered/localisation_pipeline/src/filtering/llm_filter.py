from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.config import LLMFilterConfig


@dataclass(frozen=True)
class LLMDecision:
    keep: bool
    confidence: int
    institution_present: bool
    action_present: bool
    self_contained: bool
    temporally_brittle: bool
    explanation: str
    raw_response: str


SYSTEM_PROMPT = """You are screening English sentences for a multilingual dataset about institutionally grounded localisation.
Return a JSON object with exactly these keys:
keep, confidence, institution_present, action_present, self_contained, temporally_brittle, explanation.

Definitions:
- institution_present: the sentence explicitly names an institution, public office, service centre, department, clinic, school, court, municipality, agency, or similarly grounded organisational actor.
- action_present: the sentence expresses a procedural or access-relevant action such as apply, register, submit, collect, receive, visit, qualify, or access.
- self_contained: the sentence can be understood without surrounding context.
- temporally_brittle: the sentence depends on a specific deadline, date, current-year fact, or edition-specific time reference.
- keep: true only if the sentence is a good high-precision candidate for an institutional-localisation dataset.
- confidence: integer from 1 to 5.
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


class OpenAILLMFilter:
    """LLM-assisted screening using the OpenAI Responses API.

    This implementation keeps the parsing simple and robust by asking the model to
    emit strict JSON and then validating it locally.
    """

    def __init__(self, api_key: str | None, config: LLMFilterConfig):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM filtering is enabled.")
        self.client = OpenAI(api_key=api_key)
        self.config = config

    def screen_sentence(self, sentence: str) -> LLMDecision:
        prompt = (
            "Screen this English sentence for inclusion as a high-precision institutional-localisation candidate. "
            "Return JSON only.\n\n"
            f"Sentence: {sentence}"
        )
        response = self.client.responses.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        raw = getattr(response, "output_text", "") or ""
        raw = _strip_code_fences(raw)
        payload: dict[str, Any] = json.loads(raw)
        return LLMDecision(
            keep=bool(payload["keep"]),
            confidence=int(payload["confidence"]),
            institution_present=bool(payload["institution_present"]),
            action_present=bool(payload["action_present"]),
            self_contained=bool(payload["self_contained"]),
            temporally_brittle=bool(payload["temporally_brittle"]),
            explanation=str(payload["explanation"]),
            raw_response=raw,
        )
