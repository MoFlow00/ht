"""Scoring helpers for the Discovery Hub AI-tool classification."""

from collections.abc import Iterable
import json
import re

KNOWN_AI_DOMAINS = {
    "chatgpt.com", "openai.com", "claude.ai", "gemini.google.com", "grok.com",
    "perplexity.ai", "cursor.com", "windsurf.com", "bolt.new", "v0.dev",
    "lovable.dev", "framer.ai", "replit.com", "manus.im", "groq.com",
    "cerebras.ai", "together.ai", "replicate.com", "firecrawl.dev",
    "langfuse.com", "huggingface.co", "midjourney.com", "ideogram.ai",
    "leonardo.ai", "stability.ai", "runwayml.com", "pika.art",
    "klingai.com", "pixverse.ai", "hailuoai.video", "civitai.com",
    "poe.com", "lmarena.ai", "deepseek.com", "cohere.com", "mistral.ai",
    "x.ai", "anthropic.com", "google.ai", "app.civitai.com", "aistudio.google.com",
    "sora.com", "elevenlabs.io", "suno.com", "udio.com", "runway.ml",
}

AI_STRICT_REGEX = re.compile(
    r"\b(chatgpt|openai|claude|gemini|grok|perplexity|deepseek|ollama|huggingface|midjourney|stablediffusion|cursor|windsurf|cline|copilot|manus|bolt|v0|lovable|framer|groq|cerebras|together|replicate|firecrawl|langchain|langgraph|langfuse|llm|vllm|lora|rag|mcp|agentic|coding agent|codex|diffusion|whisper|tts|embedding)\b",
    re.IGNORECASE,
)

# Generic, short model names need a company hint to count, otherwise they match everywhere.
GENERIC_MODEL_TERMS = {
    "gpt", "claude", "gemini", "command", "aya", "nova", "phi", "wan", "ray",
    "seed", "sonic", "bark", "seedream", "seedance", "kling", "gemma",
    "firefly", "playground", "kokoro", "jamba", "leonardo", "pika", "coqui",
    "chirp", "pixverse", "mixtral",
}

AI_CONTEXT_HINTS = (
    "free credits", "open weights", "model weights", "reasoning model",
    "agentic", "system prompt", "fine-tune", "fine tune", "inference",
    "prompt engineering", "ai model", "large language model", "vision model",
    "text to image", "text-to-image", "text to video", "text-to-video",
    "image generation", "video generation", "speech synthesis", "text to speech",
)

AI_SCORE_THRESHOLD = 7


def load_model_terms(catalog_path: str = "ai_models_catalog.json") -> list[str]:
    """Read vendor and model names from the catalog as whole-word matching terms."""
    try:
        with open(catalog_path, "r", encoding="utf-8") as catalog_file:
            catalog = json.load(catalog_file)
    except (OSError, ValueError):
        return []

    terms: set[str] = set()
    for companies in catalog.values():
        if not isinstance(companies, dict):
            continue
        for company, models in companies.items():
            terms.add(str(company).strip().lower())
            if isinstance(models, list):
                for model in models:
                    terms.add(str(model).strip().lower())
    return sorted(term for term in terms if len(term) >= 3 and term not in GENERIC_MODEL_TERMS)


def build_model_patterns(terms: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms]


def score_ai_link(domain: str, text: str, model_patterns: list[re.Pattern[str]]) -> int:
    """Score how strongly a link points to an AI tool. Higher means more confident."""
    domain = (domain or "").lower().replace("www.", "").strip()
    text_lower = (text or "").lower()
    score = 0

    if domain in KNOWN_AI_DOMAINS:
        score += 12
    if any(token in domain for token in ("ai", "gpt", "llm")) and not domain.startswith("mail"):
        score += 4
    if domain.endswith((".ai", ".dev")) and "aliexpress" not in domain:
        score += 2
    if AI_STRICT_REGEX.search(text_lower):
        score += 4
    for pattern in model_patterns:
        if pattern.search(text_lower):
            score += 4
            break
    if any(hint in text_lower for hint in AI_CONTEXT_HINTS):
        score += 1

    return score


def is_ai_tool(domain: str, text: str, model_patterns: list[re.Pattern[str]]) -> bool:
    return score_ai_link(domain, text, model_patterns) >= AI_SCORE_THRESHOLD


def rank_items(counter: dict, channels: dict) -> list[tuple[float, str, int, int]]:
    """Rank domains by unique channels first so one channel cannot dominate the list."""
    ranked: list[tuple[float, str, int, int]] = []
    for key, mentions in counter.items():
        unique_channels = len(channels.get(key, set()))
        if unique_channels < 1 and mentions < 2:
            continue
        score = unique_channels + mentions * 0.15
        ranked.append((score, key, mentions, unique_channels))
    ranked.sort(key=lambda row: (row[0], row[2], row[1]), reverse=True)
    return ranked
