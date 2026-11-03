"""Coach helper tests (pure functions)."""
import unittest

from backend.coach import messages_to_training_pairs


class TrainingPairTests(unittest.TestCase):
    def test_pairs_user_with_following_assistant(self):
        msgs = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "Plan a short run?"},
            {"role": "assistant", "content": "Try 20 minutes easy."},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "Rest day."},
        ]
        self.assertEqual(messages_to_training_pairs(msgs), [
            {"instruction": "Plan a short run?", "output": "Try 20 minutes easy."},
            {"instruction": "And tomorrow?", "output": "Rest day."},
        ])

    def test_keeps_entry_blocks(self):
        reply = 'Logged.\n```entry\n{ "values": { "steps": 5000 } }\n```'
        pairs = messages_to_training_pairs([
            {"role": "user", "content": "Did 5k steps"},
            {"role": "assistant", "content": reply},
        ])
        self.assertIn("```entry", pairs[0]["output"])

    def test_skips_empty_and_orphan_replies(self):
        msgs = [
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "   "},
        ]
        self.assertEqual(messages_to_training_pairs(msgs), [])


if __name__ == "__main__":
    unittest.main()
