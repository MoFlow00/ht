"""Discovery classification, ranking and conservative trend metrics."""
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
import json
import re

KNOWN_AI_DOMAINS = {
    "chatgpt.com", "openai.com", "claude.ai", "gemini.google.com", "grok.com",
    "perplexity.ai", "cursor.com", "windsurf.com", "bolt.new", "v0.dev",
    "lovable.dev", "framer.ai", "replit.com", "manus.im", "groq.com",
    "cerebras.ai", "together.ai", "replicate.com", "firecrawl.dev",
    "langfuse.com", "huggingface.co", "midjourney.com", "ideogram.ai",
    "leonardo.ai", "stability.ai", "runwayml.com", "pika.art", "klingai.com",
    "pixverse.ai", "hailuoai.video", "civitai.com", "poe.com", "lmarena.ai",
    "deepseek.com", "cohere.com", "mistral.ai", "x.ai", "anthropic.com",
    "google.ai", "app.civitai.com", "aistudio.google.com", "sora.com",
    "elevenlabs.io", "suno.com", "udio.com", "runway.ml",
}

GENERIC_MODEL_TERMS = {
    "gpt", "claude", "gemini", "command", "aya", "nova", "phi", "wan", "ray",
    "seed", "sonic", "bark", "seedream", "seedance", "kling", "gemma",
    "firefly", "playground", "kokoro", "jamba", "leonardo", "pika", "coqui",
    "chirp", "pixverse", "mixtral", "mistral", "llama", "llm", "model",
}

AI_STRICT_REGEX = re.compile(
    r"\b(chatgpt|openai|claude|gemini|grok|perplexity|deepseek|ollama|huggingface|"
    r"midjourney|stablediffusion|cursor|windsurf|cline|copilot|manus|bolt|v0|"
    r"lovable|framer|groq|cerebras|replicate|firecrawl|langchain|langgraph|"
    r"langfuse|llm|vllm|lora|rag|mcp|agentic|coding agent|codex|diffusion|"
    r"whisper|tts|embedding)\b",
    re.IGNORECASE,
)

AI_CONTEXT_HINTS = (
    "free credits", "open weights", "model weights", "reasoning model",
    "agentic", "system prompt", "fine-tune", "fine tune", "inference",
    "prompt engineering", "ai model", "large language model", "vision model",
    "text to image", "text-to-image", "text to video", "text-to-video",
    "image generation", "video generation", "speech synthesis", "text to speech",
)
AI_SCORE_THRESHOLD = 7

CATEGORY_LABELS = {
    "ai_coding_agents": "AI Coding & Agents",
    "local_ai": "Local AI & Models",
    "creative_ai": "Image, Video & Audio AI",
    "automation_api": "Automation, MCP & APIs",
    "self_hosted_devtools": "Self-hosted & Developer Tools",
    "ai_tools": "AI Tools",
    "web_utilities": "Developer Utilities",
}
CATEGORY_DOMAIN_HINTS = {
    "ai_coding_agents": {
        "cursor.com", "windsurf.com", "bolt.new", "v0.dev", "lovable.dev",
        "replit.com", "aider.chat", "cline.bot", "openhands.dev", "githubcopilot.com",
    },
    "local_ai": {
        "ollama.com", "lmstudio.ai", "jan.ai", "gpt4all.io", "localai.io",
        "ggml.ai", "llamacpp.github.io",
    },
    "creative_ai": {
        "midjourney.com", "ideogram.ai", "runwayml.com", "pika.art", "suno.com",
        "udio.com", "elevenlabs.io", "civitai.com", "stability.ai", "sora.com",
        "pixverse.ai", "hailuoai.video", "klingai.com",
    },
    "automation_api": {
        "n8n.io", "langchain.com", "langchain-ai.github.io", "langgraph.dev",
        "dify.ai", "flowiseai.com", "mcp.so", "modelcontextprotocol.io",
    },
    "self_hosted_devtools": {
        "coolify.io", "portainer.io", "docker.com", "termux.dev", "pypi.org",
        "npmjs.com", "sqlite.org",
    },
}
CATEGORY_KEYWORDS = {
    "ai_coding_agents": (
        "claude code", "codex cli", "coding agent", "code assistant", "ai coding",
        "github copilot", "qwen-coder", "deepseek-coder", "codestral", "starcoder",
        "aider", "roo code", "cline", "openhands", "swe agent", "code interpreter",
        "programming agent", "code generation",
    ),
    "local_ai": (
        "ollama", "gguf", "llama.cpp", "llamacpp", "lm studio", "local llm",
        "local model", "run locally", "self-hosted model", "open weights",
        "inference server", "vllm", "mlx", "llamafile", "localai", "gpt4all",
    ),
    "creative_ai": (
        "image generation", "text to image", "text-to-image", "video generation",
        "text to video", "text-to-video", "image ai", "video ai", "comfyui",
        "stable diffusion", "flux model", "midjourney", "elevenlabs", "text to speech",
        "speech synthesis", "ai music", "music generation", "suno ai", "udio ai",
    ),
    "automation_api": (
        "model context protocol", "mcp server", "mcp tools", "n8n workflow",
        "automation workflow", "ai automation", "openai-compatible api",
        "api endpoint", "api sdk", "webhook integration", "langgraph", "flowise",
        "dify workflow",
    ),
    "self_hosted_devtools": (
        "self-hosted", "self hosted", "docker compose", "docker container",
        "termux", "python package", "pip install", "npm package", "command line tool",
        "cli tool", "open source utility", "sqlite", "coolify", "portainer",
    ),
}
CATEGORY_TERMS_SOURCE = {
    "ai_coding_agents": ("coding_models",),
    "creative_ai": ("image_generation", "video_generation", "audio_and_speech"),
    "ai_tools": ("embedding_and_retrieval", "vision_and_multimodal", "translation_models"),
}


def parse_utc_datetime(value: object) -> datetime | None:
    """Parse supported ISO-like timestamps into timezone-aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw or raw.casefold() in {"nan", "nat", "none", "<na>"}:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
                terms.update(str(model).strip().lower() for model in models)
    return sorted(
        term for term in terms
        if len(term) >= 3 and term not in GENERIC_MODEL_TERMS
    )


def load_category_terms(catalog_path: str = "ai_models_catalog.json") -> dict[str, list[str]]:
    """Retain the catalog's category structure instead of flattening all models."""
    try:
        with open(catalog_path, "r", encoding="utf-8") as catalog_file:
            catalog = json.load(catalog_file)
    except (OSError, ValueError):
        return {}
    result: dict[str, list[str]] = {}
    for destination, source_categories in CATEGORY_TERMS_SOURCE.items():
        terms: set[str] = set()
        for source_category in source_categories:
            companies = catalog.get(source_category, {})
            if not isinstance(companies, dict):
                continue
            for models in companies.values():
                if isinstance(models, list):
                    terms.update(
                        str(model).strip().lower() for model in models
                        if len(str(model).strip()) >= 4
                        and str(model).strip().lower() not in GENERIC_MODEL_TERMS
                    )
        result[destination] = sorted(terms, key=len, reverse=True)
    return result


def build_model_patterns(terms: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms]


def _domain_matches(domain: str, known: set[str]) -> bool:
    return any(domain == item or domain.endswith("." + item) for item in known)


def score_ai_link(domain: str, text: str, model_patterns: list[re.Pattern[str]]) -> int:
    """Score how strongly a link points to an AI tool. Higher means more confident."""
    domain = (domain or "").lower().removeprefix("www.").strip().rstrip(".")
    text_lower = (text or "").lower()
    score = 12 if _domain_matches(domain, KNOWN_AI_DOMAINS) else 0
    if any(token in domain for token in ("ai", "gpt", "llm")) and not domain.startswith("mail"):
        score += 4
    if domain.endswith((".ai", ".dev")) and "aliexpress" not in domain:
        score += 2
    if AI_STRICT_REGEX.search(text_lower):
        score += 4
    if any(pattern.search(text_lower) for pattern in model_patterns):
        score += 4
    if any(hint in text_lower for hint in AI_CONTEXT_HINTS):
        score += 1
    return score


def is_ai_tool(domain: str, text: str, model_patterns: list[re.Pattern[str]]) -> bool:
    return score_ai_link(domain, text, model_patterns) >= AI_SCORE_THRESHOLD


def classify_category(
    domain: str,
    text: str,
    category_terms: dict[str, list[str]] | None = None,
    model_patterns: list[re.Pattern[str]] | None = None,
) -> dict[str, object]:
    """Return one primary category and a conservative confidence estimate."""
    clean_domain = (domain or "").lower().removeprefix("www.").strip().rstrip(".")
    context = f"{clean_domain} {text or ''}".casefold()
    context = re.sub(r"https?://\S+", " ", context)
    scores: dict[str, int] = {}
    domain_hits: set[str] = set()
    for category, domains in CATEGORY_DOMAIN_HINTS.items():
        if _domain_matches(clean_domain, domains):
            domain_hits.add(category)
    if domain_hits:
        priority = (
            "ai_coding_agents", "local_ai", "creative_ai", "automation_api",
            "self_hosted_devtools",
        )
        category = next((key for key in priority if key in domain_hits), "self_hosted_devtools")
        return {"category": category, "category_label": CATEGORY_LABELS[category], "category_confidence": 95}

    for category, terms in CATEGORY_KEYWORDS.items():
        hits = sum(
            1 for term in terms
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", context)
        )
        if hits:
            scores[category] = hits * 2
    for category, terms in (category_terms or {}).items():
        for term in terms:
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", context):
                scores[category] = scores.get(category, 0) + 2
                break
    if scores:
        priority = (
            "ai_coding_agents", "local_ai", "creative_ai", "automation_api",
            "self_hosted_devtools", "ai_tools",
        )
        category = max(priority, key=lambda key: (scores.get(key, 0), -priority.index(key)))
        confidence = min(92, 58 + scores[category] * 7)
        return {
            "category": category,
            "category_label": CATEGORY_LABELS[category],
            "category_confidence": confidence,
        }
    if is_ai_tool(clean_domain, context, model_patterns or []):
        return {"category": "ai_tools", "category_label": CATEGORY_LABELS["ai_tools"], "category_confidence": 70}
    return {
        "category": "web_utilities",
        "category_label": CATEGORY_LABELS["web_utilities"],
        "category_confidence": 45,
    }


def is_valid_github_repo(path: str) -> bool:
    """Reject GitHub navigation pages and malformed paths mistaken for repositories."""
    parts = [part.strip().strip("/") for part in (path or "").split("/") if part.strip("/")]
    if len(parts) != 2:
        return False
    owner, repo = parts
    repo = repo.removesuffix(".git")
    reserved_roots = {
        "about", "account", "apps", "collections", "codespaces", "contact",
        "customer-stories", "enterprise", "events", "explore", "features",
        "issues", "login", "marketplace", "new", "notifications", "orgs",
        "organizations", "pricing", "pulls", "readme", "search", "security",
        "sessions", "settings", "sponsors", "site", "solutions", "team",
        "topics", "trending", "users",
    }
    if owner.casefold() in reserved_roots:
        return False
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?", owner):
        return False
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?", repo):
        return False
    if repo.casefold() in reserved_roots or repo in {".", ".."}:
        return False
    if repo.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".zip")):
        return False
    return True


def calculate_trend_metrics(
    occurrences: Iterable[tuple[object, str]],
    all_channels: Iterable[str] = (),
    now: datetime | None = None,
) -> dict[str, object]:
    """
    Score link spread from Telegram post publication times, not raw view counts.
    Very young or low-sample items are explicitly kept out of Trending.
    """
    current = parse_utc_datetime(now or datetime.now(timezone.utc))
    assert current is not None
    parsed: list[tuple[datetime, str]] = []
    for timestamp, channel in occurrences:
        dt = parse_utc_datetime(timestamp)
        if dt and dt <= current + timedelta(minutes=5):
            parsed.append((dt, str(channel or "").strip().casefold()))
    latest = max((dt for dt, _ in parsed), default=None)
    first = min((dt for dt, _ in parsed), default=None)
    six_hour_start = current - timedelta(hours=6)
    day_start = current - timedelta(hours=24)
    previous_day_start = current - timedelta(hours=48)
    recent_6h = [(dt, ch) for dt, ch in parsed if dt >= six_hour_start]
    recent_24h = [(dt, ch) for dt, ch in parsed if dt >= day_start]
    previous_24h = [(dt, ch) for dt, ch in parsed if previous_day_start <= dt < day_start]
    distinct_all = {ch for ch in all_channels if ch}
    if not distinct_all:
        distinct_all = {ch for _, ch in parsed if ch}
    distinct_6h = {ch for _, ch in recent_6h if ch}

    baseline_6h = len(previous_24h) / 4
    growth_ratio = round(len(recent_6h) / baseline_6h, 2) if baseline_6h > 0 else None
    growth_percent = round((len(recent_6h) / baseline_6h - 1) * 100, 1) if baseline_6h > 0 else None
    age_minutes = max(0, int((current - latest).total_seconds() / 60)) if latest else None

    if age_minutes is not None and age_minutes <= 120 and len(recent_24h) < 3:
        status = "new"
        score = 12
        reason = "New item. More observations are needed."
    elif (
        len(recent_6h) >= 5
        and len(distinct_6h) >= 3
        and len(previous_24h) >= 3
        and growth_ratio is not None
        and growth_ratio >= 1.5
    ):
        status = "trending"
        score = min(100, 60 + len(distinct_6h) * 4 + int(min(growth_ratio, 5) * 3))
        reason = "Recent mentions accelerated against the prior 24-hour baseline."
    elif len(recent_6h) >= 3 and len(distinct_6h) >= 2 and (
        growth_ratio is None or growth_ratio >= 1.25
    ):
        status = "rising"
        score = min(85, 42 + len(distinct_6h) * 5 + int(min(growth_ratio or 1, 5) * 2))
        reason = "Several recent mentions came from independent channels."
    elif len(recent_24h) >= 10 and len(distinct_all) >= 5:
        status = "popular"
        score = min(75, 35 + len(recent_24h) + len(distinct_all))
        reason = "High recent volume across multiple channels, without enough proof of acceleration."
    elif len(recent_24h) >= 3:
        status = "active"
        score = min(55, 20 + len(recent_24h) * 2)
        reason = "Recent activity exists, but the trend sample is not strong enough."
    else:
        status = "insufficient_data"
        score = 0
        reason = "Not enough recent observations to estimate a trend."

    return {
        "trend_status": status,
        "trend_score": score,
        "trend_reason": reason,
        "mentions_6h": len(recent_6h),
        "mentions_24h": len(recent_24h),
        "mentions_previous_24h": len(previous_24h),
        "channels_6h": len(distinct_6h),
        "growth_ratio": growth_ratio,
        "growth_percent": growth_percent,
        "post_age_minutes": age_minutes,
        "earliest_sampled_post_at": first.isoformat() if first else "",
        "latest_post_at": latest.isoformat() if latest else "",
        "trend_data_sufficient": len(previous_24h) >= 3 and len(recent_24h) >= 3,
    }


def rank_items(counter: dict, channels: dict) -> list[tuple[float, str, int, int]]:
    """Rank discoveries by a blend of unique-channel spread and repeated mentions."""
    ranked: list[tuple[float, str, int, int]] = []
    for key, mentions in counter.items():
        unique_channels = len(channels.get(key, set()))
        if unique_channels < 1 and mentions < 2:
            continue
        score = unique_channels + mentions * 0.15
        ranked.append((score, key, mentions, unique_channels))
    ranked.sort(key=lambda row: (row[0], row[2], row[1]), reverse=True)
    return ranked


def rank_top_domains(counter: dict, channels: dict, limit: int = 100) -> list[dict]:
    """Blend frequency and channel diversity, avoiding dominance by one source."""
    ranked: list[dict] = []
    for domain, mentions in counter.items():
        if mentions < 1:
            continue
        unique_channels = len(channels.get(domain, set()))
        score = mentions * 0.6 + unique_channels * 0.4
        ranked.append({
            "domain": domain,
            "mentions": mentions,
            "unique_channels": unique_channels,
            "score": round(score, 2),
        })
    ranked.sort(key=lambda row: (row["score"], row["mentions"], len(row["domain"])), reverse=True)
    return ranked[:limit]
