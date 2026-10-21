import unittest

from explorer_scoring import (
    build_model_patterns,
    is_ai_tool,
    load_model_terms,
    rank_items,
    score_ai_link,
)


class ExplorerScoringTests(unittest.TestCase):
    def setUp(self):
        self.patterns = build_model_patterns(load_model_terms("ai_models_catalog.json"))

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


if __name__ == "__main__":
    unittest.main()
