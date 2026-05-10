"""Pickle ``predict_proba_point_only`` vs JSON snapshot at same ``as_of_date`` (strict)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_artifacts  # noqa: E402
from src.export.json_inference import predict_proba_snapshot  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402

from tests.harness_skip import (  # noqa: E402
    HAS_HARNESS_MODEL,
    harness_model_path,
    print_harness_integration_preamble,
)
from tests.parity_helpers import assert_point_probs_match_pkl  # noqa: E402


def setUpModule() -> None:
    print_harness_integration_preamble(
        module="tests.test_artifact_parity",
        description=(
            "Strict parity: pickle predict_proba_point_only vs JSON-only predict_proba_snapshot "
            "(same fight_date == artifact as_of_date)."
        ),
    )


@unittest.skipUnless(
    HAS_HARNESS_MODEL,
    "No pickle for harness (see stderr banner: data/model.pkl, MMA_HARNESS_MODEL, fixture).",
)
class TestPickleVsJsonParity(unittest.TestCase):
    def test_exported_bundle_matches_predict_proba_point_only(self) -> None:
        from datetime import date

        model_path = harness_model_path()
        assert model_path is not None
        predictor = MMAPredictor.load(model_path)
        if predictor.fights:
            d_asof = predictor.fights[-1].fight_date
        else:
            d_asof = date.today()

        triples: list[tuple[str, str, object]] = []
        for fr in reversed(predictor.fights):
            if len(triples) >= 15:
                break
            triples.append((fr.fighter_a_id, fr.fighter_b_id, fr.weight_class))

        if not triples:
            self.skipTest("model has no fights to sample parity triples")

        print(
            f"[parity] Pickle: {model_path}\n"
            f"[parity] Snapshot as_of_date (fight_date for both paths): {d_asof.isoformat()}\n"
            f"[parity] Matchups to compare: {len(triples)} (max 15, most recent fights first)",
            flush=True,
            file=sys.stderr,
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            print("[parity] Exporting four JSON files to temp dir (full model state)...", flush=True, file=sys.stderr)
            export_artifacts.export_all(predictor, out, as_of=d_asof)

            for i, (a_id, b_id, wc) in enumerate(triples):
                with self.subTest(i=i, a=a_id, b=b_id, wc=wc, d=d_asof):
                    label = (
                        f"{i + 1}/{len(triples)}  fighter_a={a_id}  fighter_b={b_id}  wc={wc!s}  date={d_asof}"
                    )
                    print(f"[parity] Subtest {label}", flush=True, file=sys.stderr)
                    p_pkl = predictor.predict_proba_point_only(a_id, b_id, wc, d_asof)
                    p_json = predict_proba_snapshot(out, a_id, b_id, wc, d_asof)
                    assert_point_probs_match_pkl(
                        p_pkl,
                        p_json,
                        context=f"triple_index={i} {a_id} vs {b_id} {wc} @ {d_asof}",
                    )
                    print(
                        "[parity]   OK: 6-class point probs identical (pickle vs JSON snapshot).",
                        flush=True,
                        file=sys.stderr,
                    )

        print(f"[parity] DONE: all {len(triples)} subtests matched.", flush=True, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
