"""
Lightweight unit tests for core AEC-Triad helpers.

Run from repo root:
    python -m unittest tests.test_core_analyzers -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow importing modules from the repo root without an install step.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_signal_utils import extract_engagement_level
from performance_analyzer import PerformanceAnalyzer
from temporal_segmenter import TemporalSegmenter, TemporalSegmentType


def _frame(ts: float, action: str, engagement: str = "ENGAGED", collab: float = 0.2, frame: int = 0):
    return {
        "timestamp": ts,
        "action_name": action,
        "engagement_level": engagement,
        "collaboration_score": collab,
        "frame_number": frame,
    }


class TestEngagementExtraction(unittest.TestCase):
    def test_phone_use_is_disengaged(self):
        level = extract_engagement_level("The operator is using mobile phone and looking away.")
        self.assertEqual(level, "DISENGAGED")

    def test_assembly_focus_is_engaged(self):
        level = extract_engagement_level(
            "Operator is carefully assembling and aligning the component."
        )
        self.assertIn(level, {"ENGAGED", "HIGHLY_ENGAGED"})


class TestPerformanceAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PerformanceAnalyzer()

    def test_baseline_score_in_range(self):
        metrics = self.analyzer.analyze_performance(
            analysis="Operator places the part cleanly.",
            engagement_level="ENGAGED",
            hand_usage="BOTH",
            action_name="place part",
        )
        self.assertTrue(30 <= metrics["performance_score"] <= 95)
        self.assertFalse(metrics["fatigue_detected"])

    def test_safety_language_lowers_score(self):
        safe = self.analyzer.analyze_performance(
            "Clean placement with proper grip.",
            "ENGAGED",
            "BOTH",
            "place part",
        )
        # Need several safety pattern hits — threshold is relatively strict.
        risky = self.analyzer.analyze_performance(
            "Unsafe dangerous risk hazard careless reckless improper grip wrong tool.",
            "ENGAGED",
            "BOTH",
            "place part",
        )
        self.assertTrue(risky["safety_concerns"])
        self.assertLess(risky["performance_score"], safe["performance_score"])


class TestTemporalSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = TemporalSegmenter(min_segment_duration=2.0)

    def test_classify_preparation_and_execution(self):
        prep = self.segmenter._classify_segment_type(
            _frame(0.0, "take screwdriver"), None
        )
        exec_ = self.segmenter._classify_segment_type(
            _frame(1.0, "screw panel", collab=0.2), None
        )
        collab = self.segmenter._classify_segment_type(
            _frame(2.0, "assemble joint", collab=0.8), None
        )
        self.assertEqual(prep, TemporalSegmentType.PREPARATION)
        self.assertEqual(exec_, TemporalSegmentType.EXECUTION)
        self.assertEqual(collab, TemporalSegmentType.COLLABORATION)

    def test_final_segment_is_not_dropped(self):
        # Same action across enough duration should produce one flushed segment.
        results = [
            _frame(0.0, "screw bolt", frame=0),
            _frame(1.0, "screw bolt", frame=1),
            _frame(2.5, "screw bolt", frame=2),
        ]
        segments = self.segmenter.segment_assembly_process(results)
        self.assertGreaterEqual(len(segments), 1)
        self.assertEqual(segments[-1].segment_type, TemporalSegmentType.EXECUTION)
        self.assertGreaterEqual(segments[-1].duration, 2.0)


if __name__ == "__main__":
    unittest.main()
