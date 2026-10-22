import unittest
from datetime import datetime, timedelta, timezone

from explorer_scoring import (
    build_model_patterns,
    calculate_trend_metrics,
    classify_category,
    is_ai_tool,
    is_valid_github_repo,
    load_category_terms,
    load_model_terms,
    rank_items,
    score_ai_link,
)


class ExplorerScoringTests(unittest.TestCase):
    def setUp(self):
        self.patterns = build_model_patterns(load_model_terms("ai_models_catalog.json"))
        self.category_terms = load_category_terms("ai_models_catalog.json")
        self.now = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)

    def test_catalog_terms_are_loaded(self):
        terms = load_model_terms("ai_models_catalog.json")
        self.assertIn("qwen-coder", terms)
        self.assertIn("deepseek-r1", terms)
        self.assertNotIn("gpt", terms)
        self.assertNotIn("claude", terms)

    def test_known_ai_domain_is_classified(self):
        self.assertTrue(is_ai_tool("perplexity.ai", "Ask anything", self.patterns))
        self.assertTrue(is_ai_tool("bolt.new", "Build an app", self.patterns))

    def test_catalog_model_name_scores_ai(self):
        self.assertTrue(is_ai_tool("example.dev", "Qwen-Coder is now open weights", self.patterns))
        self.assertTrue(is_ai_tool("example.dev", "Try DeepSeek-R1 locally", self.patterns))

    def test_plain_utility_domain_is_not_ai(self):
        self.assertFalse(is_ai_tool("jsonformatter.org", "Pretty print JSON online for free", self.patterns))
        self.assertFalse(is_ai_tool("example.com", "A normal news post about nothing", self.patterns))

    def test_generic_model_name_alone_does_not_trigger(self):
        self.assertFalse(is_ai_tool("news.example", "The Nova stadium reopened", self.patterns))

    def test_ranking_prefers_many_unique_channels(self):
        counter = {"a.com": 10, "b.com": 3}
        channels = {"a.com": {"one"}, "b.com": {"one", "two", "three"}}
        ranked = rank_items(counter, channels)
        self.assertEqual(ranked[0][1], "b.com")

    def test_score_is_stable_for_known_domain(self):
        score = score_ai_link("chatgpt.com", "hello", self.patterns)
        self.assertGreaterEqual(score, 12)

    def test_coding_agent_category(self):
        category = classify_category(
            "example.com", "Claude Code is a coding agent for software repositories",
            self.category_terms, self.patterns,
        )
        self.assertEqual(category["category"], "ai_coding_agents")
        self.assertGreaterEqual(category["category_confidence"], 58)

    def test_local_ai_category(self):
        category = classify_category(
            "example.com", "Run Ollama with a GGUF model locally",
            self.category_terms, self.patterns,
        )
        self.assertEqual(category["category"], "local_ai")

    def test_automation_category(self):
        category = classify_category(
            "example.com", "Model Context Protocol MCP server and n8n workflow",
            self.category_terms, self.patterns,
        )
        self.assertEqual(category["category"], "automation_api")

    def test_known_domain_category(self):
        category = classify_category("cursor.com", "An unrelated sample", self.category_terms, self.patterns)
        self.assertEqual(category["category"], "ai_coding_agents")
        self.assertEqual(category["category_confidence"], 95)

    def test_github_navigation_paths_are_rejected(self):
        for path in ("topics/python", "marketplace/copilot", "settings/profile", "user/repo.png"):
            with self.subTest(path=path):
                self.assertFalse(is_valid_github_repo(path))

    def test_normal_github_repositories_are_accepted(self):
        self.assertTrue(is_valid_github_repo("pytorch/pytorch"))
        self.assertTrue(is_valid_github_repo("MoFlow00/ht"))
        self.assertTrue(is_valid_github_repo("owner/repo.git"))

    def test_new_item_is_not_mislabeled_as_trending(self):
        metrics = calculate_trend_metrics(
            [(self.now - timedelta(minutes=5), "one")], {"one"}, self.now,
        )
        self.assertEqual(metrics["trend_status"], "new")
        self.assertEqual(metrics["mentions_6h"], 1)
        self.assertFalse(metrics["trend_data_sufficient"])

    def test_two_posts_from_one_channel_are_not_a_trend(self):
        occurrences = [
            (self.now - timedelta(minutes=15), "same"),
            (self.now - timedelta(minutes=10), "same"),
        ]
        metrics = calculate_trend_metrics(occurrences, {"same"}, self.now)
        self.assertNotEqual(metrics["trend_status"], "trending")
        self.assertNotEqual(metrics["trend_status"], "rising")

    def test_growth_requires_baseline_and_channel_diversity(self):
        previous = [(self.now - timedelta(hours=30 + i), f"old{i}") for i in range(3)]
        recent = [
            (self.now - timedelta(minutes=10), "alpha"),
            (self.now - timedelta(minutes=20), "beta"),
            (self.now - timedelta(minutes=30), "gamma"),
            (self.now - timedelta(minutes=40), "alpha"),
            (self.now - timedelta(minutes=50), "beta"),
        ]
        metrics = calculate_trend_metrics(previous + recent, {"old0", "old1", "old2", "alpha", "beta", "gamma"}, self.now)
        self.assertEqual(metrics["trend_status"], "trending")
        self.assertGreater(metrics["growth_ratio"], 1.5)

    def test_missing_dates_report_insufficient_data(self):
        metrics = calculate_trend_metrics([("", "one"), ("not-a-date", "two")], {"one", "two"}, self.now)
        self.assertEqual(metrics["trend_status"], "insufficient_data")
        self.assertIsNone(metrics["growth_ratio"])


if __name__ == "__main__":
    unittest.main()
