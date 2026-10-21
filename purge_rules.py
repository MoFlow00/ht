"""Exact whole-word/phrase matching helpers for destructive data-cleanup rules."""

from collections.abc import Iterable, Mapping
import re
import unicodedata


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_username(value: object) -> str:
    normalized = "" if value is None else str(value).strip().lstrip("@").casefold()
    return "" if normalized in {"nan", "<na>", "nat"} else normalized


def build_matcher(terms: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile case-insensitive exact word/phrase rules with Unicode word boundaries."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()

    for raw in terms:
        if not isinstance(raw, str):
            raise ValueError("Every purge term must be a string")
        term = raw.strip()
        normalized = _normalize(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        # Normalize spacing between words, while keeping punctuation literal.
        body = r"\s+".join(re.escape(part) for part in normalized.split())
        word_chars = r"\w\u0300-\u036f\u064b-\u065f\u0670"
        prefix = rf"(?<![{word_chars}])"
        suffix = rf"(?![{word_chars}])"
        compiled.append((term, re.compile(prefix + body + suffix, re.UNICODE)))

    return compiled


def match_terms(text: object, rules: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    """Return configured terms found as whole words/phrases, never as substrings."""
    if text is None:
        return []
    normalized_text = _normalize(str(text))
    return [term for term, pattern in rules if pattern.search(normalized_text)]


def channels_to_purge(
    posts: Iterable[Mapping[str, object]],
    rules: list[tuple[str, re.Pattern[str]]],
    *,
    include_nameless: bool = False,
) -> set[str] | tuple[set[str], int]:
    """Collect all channels with at least one matching post; optionally count nameless hits."""
    blocked: set[str] = set()
    nameless_matches = 0
    for post in posts:
        if not match_terms(post.get("text", ""), rules):
            continue
        username = normalize_username(post.get("username", ""))
        if username:
            blocked.add(username)
        else:
            nameless_matches += 1
    if include_nameless:
        return blocked, nameless_matches
    return blocked
