"""All records here are synthetic identity/join plumbing, not research evidence."""
from dataclasses import replace
import unittest

from em_geometry.records import (ResponseIdentity, JudgmentIdentity, unique_index,
                                 strict_join, align_legacy_pairs)


class RecordsTests(unittest.TestCase):
    def setUp(self):
        self.response = ResponseIdentity("run-a", "checkpoint-a", 1, "prompt-a", 0)

    def test_composite_response_identity(self):
        variants = [self.response,
                    replace(self.response, run="run-b"),
                    replace(self.response, checkpoint="checkpoint-b"),
                    replace(self.response, training_seed=2),
                    replace(self.response, prompt="prompt-b"),
                    replace(self.response, response_sample=1)]
        self.assertEqual(len(set(variants)), 6)

    def test_judge_and_pass_are_separate(self):
        base = JudgmentIdentity(self.response, "judge-a", "pass-1")
        self.assertNotEqual(base, replace(base, judge="judge-b"))
        self.assertNotEqual(base, replace(base, pass_id="pass-2"))
        self.assertNotEqual(base, replace(base, response=replace(self.response, response_sample=1)))

    def test_invalid_identity_fails(self):
        for kwargs in ({"run": ""}, {"response_sample": -1}, {"training_seed": True}):
            with self.assertRaises(ValueError):
                replace(self.response, **kwargs)

    def test_exact_join_preserves_left_order(self):
        self.assertEqual(strict_join([2, 1], [1, 2], lambda x: x), [(2, 2), (1, 1)])

    def test_duplicates_and_missing_do_not_silently_drop(self):
        with self.assertRaises(ValueError):
            unique_index([self.response, self.response], lambda x: x)
        for left, right in [([1], [2]), ([1, 1], [1]), ([1], [1, 1])]:
            with self.assertRaises(ValueError):
                strict_join(left, right, lambda x: x)

    def test_legacy_alignment_sorted_and_text_exact(self):
        reference = [{"idx": 0, "prompt": "p0", "response": "r0"},
                     {"idx": 1, "prompt": "p1", "response": "r1"}]
        self.assertEqual(align_legacy_pairs(reference, reference[::-1], expected_count=2), reference)
        for field in ("prompt", "response", "idx"):
            bad = [dict(r) for r in reference]
            bad[0][field] = "different"
            with self.assertRaises(ValueError):
                align_legacy_pairs(reference, bad, expected_count=2)
        with self.assertRaises(ValueError):
            align_legacy_pairs(reference, reference, expected_count=3)


if __name__ == "__main__":
    unittest.main()
