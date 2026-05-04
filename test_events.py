"""Unit tests for api/events.py — stdlib only, no network.

Run: python test_events.py
"""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("events", ROOT / "api" / "events.py")
events = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(events)


class TestMatchRules(unittest.TestCase):
    def test_launchpool_match(self):
        m = events.match_rules("Binance Launchpool: New Token X")
        self.assertIsNotNone(m)
        self.assertIn("launchpool", m["matched"])

    def test_simple_earn_match(self):
        m = events.match_rules("Simple Earn Promotion Bonus")
        self.assertIsNotNone(m)
        self.assertIn("simple earn", m["matched"])
        self.assertIn("promotion", m["matched"])
        self.assertIn("bonus", m["matched"])

    def test_deposit_event(self):
        m = events.match_rules("Deposit BTC and Earn Reward")
        self.assertIsNotNone(m)
        self.assertIn("deposit", m["matched"])
        self.assertIn("earn", m["matched"])

    def test_word_boundary_learn_excluded(self):
        # "earn" must not match inside "learn"
        m = events.match_rules("Learn How to Trade")
        self.assertIsNone(m)

    def test_word_boundary_yearn_excluded(self):
        m = events.match_rules("Yearn Protocol Listed")
        self.assertIsNone(m)

    def test_exclude_overrides_include(self):
        # "Trading Competition" excluded even if "bonus" included
        m = events.match_rules("Trading Competition with Bonus Reward")
        self.assertIsNone(m)

    def test_apy_hint_extraction(self):
        m = events.match_rules("Earn 12.5% APY on BNB")
        self.assertIsNotNone(m)
        self.assertEqual(m["apy_hint"], "12.5")

    def test_apy_hint_none_when_no_pct(self):
        m = events.match_rules("Earn rewards on BNB staking")
        self.assertIsNotNone(m)
        self.assertIsNone(m["apy_hint"])

    def test_no_include_keyword(self):
        m = events.match_rules("New trading pair listed: ABC/USDT")
        self.assertIsNone(m)

    def test_empty_title(self):
        self.assertIsNone(events.match_rules(""))


class TestToIso(unittest.TestCase):
    def test_int_milliseconds(self):
        # 2009-02-13T23:31:30+00:00
        out = events.to_iso(1234567890000)
        self.assertTrue(out.startswith("2009-02-13T"))

    def test_bool_true_returns_none(self):
        # bool is a subclass of int — must NOT be treated as timestamp.
        self.assertIsNone(events.to_iso(True))

    def test_bool_false_returns_none(self):
        self.assertIsNone(events.to_iso(False))

    def test_string_passthrough(self):
        self.assertEqual(events.to_iso("2026-05-04T00:00:00Z"), "2026-05-04T00:00:00Z")

    def test_none_returns_none(self):
        self.assertIsNone(events.to_iso(None))

    def test_list_returns_none(self):
        self.assertIsNone(events.to_iso([]))


class TestExtractArticles(unittest.TestCase):
    def test_nested_catalogs(self):
        payload = {"data": {"catalogs": [
            {"catalogName": "News", "articles": [{"title": "a"}, {"title": "b"}]},
            {"catalogName": "Activities", "articles": [{"title": "c"}]},
        ]}}
        flat = events.extract_articles(payload)
        self.assertEqual(len(flat), 3)
        self.assertEqual(flat[0]["_catalogName"], "News")
        self.assertEqual(flat[2]["_catalogName"], "Activities")

    def test_flat_articles_fallback(self):
        payload = {"data": {"articles": [{"title": "a"}]}}
        flat = events.extract_articles(payload)
        self.assertEqual(len(flat), 1)

    def test_empty_payload(self):
        self.assertEqual(events.extract_articles({}), [])

    def test_data_null(self):
        self.assertEqual(events.extract_articles({"data": None}), [])

    def test_skips_non_dict_articles(self):
        payload = {"data": {"catalogs": [
            {"catalogName": "X", "articles": [{"title": "ok"}, "junk", None, 42]}
        ]}}
        flat = events.extract_articles(payload)
        self.assertEqual(len(flat), 1)


class TestBuildUrl(unittest.TestCase):
    def test_with_code(self):
        url = events._build_url({"code": "abc-123-xyz"})
        self.assertEqual(url, "https://www.binance.com/en/support/announcement/abc-123-xyz")

    def test_empty_code(self):
        self.assertEqual(events._build_url({"code": ""}), "")

    def test_none_code(self):
        self.assertEqual(events._build_url({"code": None}), "")

    def test_int_code_rejected(self):
        # articleId-only fallback was removed — numeric codes don't form valid URLs.
        self.assertEqual(events._build_url({"code": 123}), "")

    def test_articleId_only_returns_empty(self):
        self.assertEqual(events._build_url({"articleId": 999}), "")

    def test_special_chars_encoded(self):
        url = events._build_url({"code": "a/b c"})
        self.assertIn("a%2Fb%20c", url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
