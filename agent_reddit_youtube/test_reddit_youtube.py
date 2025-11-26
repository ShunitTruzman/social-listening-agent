import unittest
from main import detect_entities, safe_json_load, agg_sentiment_per_field

class TestSocialListeningAgent(unittest.TestCase):

    def test_detect_entities(self):
        """Entities must be detected correctly."""
        self.assertEqual(detect_entities("Taboola is good"), ["Taboola"])
        self.assertEqual(detect_entities("I like Realize app"), ["Realize"])
        self.assertEqual(
            detect_entities("Taboola and Realize both appear"),
            ["Taboola", "Realize"]
        )
        self.assertEqual(detect_entities("Nothing here"), [])

    def test_safe_json_load(self):
        """safe_json_load should parse valid or wrapped JSON."""
        raw = '{"a": 1, "b": 2}'
        self.assertEqual(safe_json_load(raw), {"a": 1, "b": 2})

        wrapped = "RANDOM TEXT {\"x\": 5} MORE TEXT"
        self.assertEqual(safe_json_load(wrapped), {"x": 5})

    def test_agg_sentiment_per_field(self):
        """Sentiment aggregation must sum correctly."""
        records = [
            {
                "entities": ["Taboola"],
                "fields": [{"field": "performance", "sentiment": "positive"}]
            },
            {
                "entities": ["Taboola"],
                "fields": [{"field": "performance", "sentiment": "negative"}]
            },
            {
                "entities": ["Realize"],
                "fields": [{"field": "ux", "sentiment": "neutral"}]
            },
        ]

        aggregated = agg_sentiment_per_field(records)

        # Taboola
        self.assertEqual(aggregated["Taboola"]["performance"]["positive"], 1)
        self.assertEqual(aggregated["Taboola"]["performance"]["negative"], 1)

        # Realize
        self.assertEqual(aggregated["Realize"]["ux"]["neutral"], 1)


if __name__ == "__main__":
    unittest.main()
