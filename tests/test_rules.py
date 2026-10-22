"""정규화의 수학과 규칙의 문맥 경계를 확인한다."""

import unittest
from extract import extract, korean_amount, mask_pii
from sentiment import analyze, aspects


class RulesTests(unittest.TestCase):
    def test_korean_large_unit_amount(self):
        self.assertEqual(korean_amount("1억 2천만원"), 120000000)
        self.assertEqual(korean_amount("2만 5천원"), 25000)
        self.assertEqual(korean_amount("1.5만원"), 15000)
        with self.assertRaises(ValueError):
            korean_amount("1,2,3원")

    def test_calendar_and_phone(self):
        self.assertEqual(extract("2024-02-30"), [])
        self.assertEqual(extract("+82-10-0000-1001")[0]["value"], "010-0000-1001")
        self.assertEqual(extract("2024년 2월 29일")[0]["value"], "2024-02-29")

    def test_dollar_cents_and_url_parentheses(self):
        self.assertEqual(
            extract("$12.50.")[0]["value"], {"amount_minor": 1250, "currency": "USD"}
        )
        self.assertEqual(
            extract("(https://example.test/a(b)).")[0]["value"],
            "https://example.test/a(b)",
        )

    def test_negation_intensifier_and_double_negation(self):
        base = analyze("good")["score"]
        self.assertGreater(base, 0)
        self.assertLess(analyze("not good")["score"], 0)
        self.assertGreater(analyze("very good")["score"], base)
        self.assertGreater(analyze("not not good")["score"], 0)
        self.assertGreater(analyze("I can't dislike it.")["score"], 0)

    def test_korean_required_patterns(self):
        self.assertLess(analyze("좋지 않다")["score"], 0)
        self.assertGreater(analyze("매우 좋다")["score"], analyze("좋다")["score"])
        self.assertGreater(analyze("안 좋지 않다")["score"], 0)

    def test_negation_boundary_and_numeric_grouping(self):
        self.assertGreater(analyze("not bad but good")["contributions"][-1]["score"], 0)
        self.assertGreater(analyze("편안한 좋은 연기")["score"], 0)
        self.assertEqual(extract("$1,2,3"), [])
        self.assertEqual(extract("1,2,3원"), [])

    def test_aspect_opposite_clauses(self):
        result = aspects("The story is excellent but the acting is terrible.")[
            "aspects"
        ]
        self.assertEqual(result["story"]["sentiment"], "positive")
        self.assertEqual(result["acting"]["sentiment"], "negative")
        result = aspects("음식은 매우 좋다. 가격은 비싸다.")["aspects"]
        self.assertEqual(result["food"]["sentiment"], "positive")
        self.assertEqual(result["price"]["sentiment"], "negative")

    def test_mask_synthetic_identifiers(self):
        source = "000000-1000000, 0000 0000 0000 0000, 서울특별시 예시구 샘플로 123, user@example.test"
        result = mask_pii(source)
        self.assertIn("[RESIDENT]", result)
        self.assertIn("[CARD]", result)
        self.assertIn("[ADDRESS]", result)
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("1000000", result)
        self.assertNotIn("샘플로", result)
