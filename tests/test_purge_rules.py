import unittest

from purge_rules import build_matcher, match_terms, channels_to_purge, normalize_username


class ExactPurgeRuleTests(unittest.TestCase):
    def setUp(self):
        self.rules = build_matcher(["termux", "api key", "إعلان ممنوع"])

    def test_matches_whole_english_word(self):
        self.assertEqual(match_terms("Need TERMUX now", self.rules), ["termux"])

    def test_does_not_match_inside_another_word(self):
        self.assertEqual(match_terms("I use termuxer and premiumtermux", self.rules), [])

    def test_matches_phrase_with_flexible_whitespace_and_punctuation(self):
        self.assertEqual(match_terms("Leaked API   KEY!", self.rules), ["api key"])

    def test_does_not_match_partial_phrase_word(self):
        self.assertEqual(match_terms("api keys are available", self.rules), [])

    def test_underscore_is_part_of_a_word(self):
        self.assertEqual(match_terms("termux_tools", self.rules), [])

    def test_punctuation_ended_term_is_not_a_prefix_of_a_word(self):
        rules = build_matcher(["C++"])
        self.assertEqual(match_terms("I write C++ daily", rules), ["C++"])
        self.assertEqual(match_terms("C++language", rules), [])

    def test_matches_arabic_phrase_only_at_word_boundaries(self):
        self.assertEqual(match_terms("هذا إعلان ممنوع!", self.rules), ["إعلان ممنوع"])
        self.assertEqual(match_terms("إعلان ممنوعة", self.rules), [])
        self.assertEqual(match_terms("إعلان ممنوعٌ", self.rules), [])

    def test_normalizes_at_sign_case_and_missing_values(self):
        self.assertEqual(normalize_username(" @Channel_A "), "channel_a")
        self.assertEqual(normalize_username(float("nan")), "")

    def test_channel_collection_flags_every_post_from_a_matching_channel(self):
        posts = [
            {"username": "Channel_A", "text": "Normal announcement"},
            {"username": "channel_b", "text": "An API key leaked"},
            {"username": "CHANNEL_B", "text": "A harmless post"},
            {"username": "channel_c", "text": "premiumtermux"},
        ]
        self.assertEqual(channels_to_purge(posts, self.rules), {"channel_b"})

    def test_matching_post_without_channel_is_not_lost_from_audit_result(self):
        posts = [{"username": "", "text": "termux"}]
        blocked, nameless_matches = channels_to_purge(posts, self.rules, include_nameless=True)
        self.assertEqual(blocked, set())
        self.assertEqual(nameless_matches, 1)


if __name__ == "__main__":
    unittest.main()
