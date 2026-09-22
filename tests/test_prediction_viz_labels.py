"""
Legend labels in ``plot_prediction_three_viz`` must match the model's own outcome encoding.

The win side (classes 0-2) and the lose side (classes 3-5) carry *different* method orders —
KO/TKO, Submission, Decision versus Decision, KO/TKO, Submission. Indexing one hand-written
tuple with ``outcome_index % 3`` (or ``- 3``) therefore rotates every corner-B label without
raising, so these assertions derive the truth from ``_OUTCOME_TO_CLASS`` / ``CLASS_LABELS``
rather than restating the order.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.plot_prediction_three_viz import (  # noqa: E402
    _class_is_corner_a,
    _class_method,
    _outcome_sentence,
    _thin_ci_caption,
)
from src.data.schema import ResultMethod  # noqa: E402
from src.model.regression import CLASS_LABELS, N_CLASSES, _OUTCOME_TO_CLASS  # noqa: E402

#: Method wording used by the charts for each :class:`ResultMethod` reaching a class index.
_METHOD_WORDING = {
    ResultMethod.KO_TKO: "KO/TKO",
    ResultMethod.SUBMISSION: "Submission",
    ResultMethod.UNANIMOUS_DECISION: "Decision",
    ResultMethod.SPLIT_DECISION: "Decision",
    ResultMethod.MAJORITY_DECISION: "Decision",
}

_A = "Alpha Ant"
_B = "Beta Bee"


class TestOutcomeLabelsMatchClassEncoding(unittest.TestCase):
    def test_every_class_index_labels_the_corner_and_method_it_encodes(self) -> None:
        """For each ``(fighter_won, result_method) -> class``, the label must name that outcome."""
        self.assertEqual(len(CLASS_LABELS), N_CLASSES)
        covered: set[int] = set()

        for (won, method), idx in _OUTCOME_TO_CLASS.items():
            covered.add(idx)
            expected_method = _METHOD_WORDING[method]
            expected_corner = _A if won else _B

            self.assertEqual(
                _outcome_sentence(_A, _B, idx),
                f"{expected_corner} by {expected_method}",
                msg=f"class {idx} ({CLASS_LABELS[idx]}) mislabelled for won={won} method={method}",
            )
            self.assertEqual(_class_method(idx), expected_method)
            self.assertIs(_class_is_corner_a(idx), won)

        self.assertEqual(covered, set(range(N_CLASSES)), "every class index must be exercised")

    def test_thin_ci_caption_uses_the_same_corner_and_method(self) -> None:
        """The off-bar short caption shares the encoding, not a second hand-written tuple."""
        short = {"KO/TKO": "KO/TKO", "Submission": "Sub", "Decision": "Dec"}
        for (won, method), idx in _OUTCOME_TO_CLASS.items():
            expected_last_name = "Ant" if won else "Bee"
            self.assertEqual(
                _thin_ci_caption(_A, _B, idx, 4, 11),
                f"{expected_last_name} · {short[_METHOD_WORDING[method]]}  [4–11%]",
                msg=f"class {idx} ({CLASS_LABELS[idx]}) caption mislabelled",
            )

    def test_win_and_lose_sides_do_not_share_a_method_order(self) -> None:
        """Guards the regression directly: rotating one tuple would make these two equal."""
        win_side = tuple(_class_method(i) for i in range(3))
        lose_side = tuple(_class_method(i) for i in range(3, N_CLASSES))
        self.assertEqual(win_side, ("KO/TKO", "Submission", "Decision"))
        self.assertEqual(lose_side, ("Decision", "KO/TKO", "Submission"))
        self.assertNotEqual(win_side, lose_side)


if __name__ == "__main__":
    unittest.main()
