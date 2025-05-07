# Unit tests
import unittest
from prepare_rl_set_from_traces import process_trace_rewards

class TestRewardProcessing(unittest.TestCase):
    def setUp(self):
        self.sample_trace1 = [
            {"type": "orientation", "score": 2.5, "index": 0},
            {"type": "action", "score": 1.8, "index": 1},
            {"type": "expectation", "score": 3.0, "index": 2}
        ]
        
        self.sample_trace2 = [
            {"type": "orientation", "score": 3.5, "index": 0},
            {"type": "action", "score": 2.0, "index": 1, 
             "outcome": {"error": "AssertionError"}},
            {"type": "error", "index": 2},
            {"type": "debug", "score": 2.5, "index": 3},
            {"type": "action", "score": 2.2, "index": 4},
            {"type": "expectation", "score": 3.0, "index": 5}
        ]
        
        self.sample_trace3 = [
            {"type": "orientation", "score": 2.0, "index": 0},
            {"type": "action", "score": 1.8, "index": 1, 
             "outcome": {"error": "TypeError"}},
            {"type": "error", "index": 2},
            {"type": "debug", "score": 2.0, "index": 3},
            {"type": "action", "score": 1.9, "index": 4,
             "outcome": {"error": "KeyError"}},
            {"type": "error", "index": 5},
            {"type": "debug", "score": 1.8, "index": 6},
            {"type": "action", "score": 2.1, "index": 7,
             "outcome": {"error": "AssertionError"}},
            {"type": "error", "index": 8},
            {"type": "debug", "score": 2.2, "index": 9},
            {"type": "orientation", "score": 2.5, "index": 10}
        ]

    def test_successful_action_chain(self):
        processed = process_trace_rewards(self.sample_trace1)
        # Orientation should get +0.1 (2.5 -> 2.6)
        self.assertAlmostEqual(processed[0]["score"], 2.6)
        # Action should keep original score minus no error penalty
        self.assertAlmostEqual(processed[1]["score"], 1.8)

    def test_mixed_success_chain(self):
        processed = process_trace_rewards(self.sample_trace2)
        self.assertAlmostEqual(processed[0]["score"], 3.25)
        self.assertAlmostEqual(processed[1]["score"], 1.75)
        self.assertAlmostEqual(processed[3]["score"], 2.6)
        self.assertAlmostEqual(processed[4]["score"], 2.2)

    def test_triple_failure_chain(self):
        processed = process_trace_rewards(self.sample_trace3)
        self.assertAlmostEqual(processed[0]["score"], 1.9)
        self.assertAlmostEqual(processed[3]["score"], 1.9)
        self.assertAlmostEqual(processed[6]["score"], 1.7)
        self.assertAlmostEqual(processed[1]["score"], 1.3)
        self.assertAlmostEqual(processed[4]["score"], 1.4)
        self.assertAlmostEqual(processed[7]["score"], 1.85)

if __name__ == "__main__":
    unittest.main()
