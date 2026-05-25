"""UFCStats scraper guards against empty overwrites when the index is a bot challenge."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from src.data.ufcstats_scraper import (
    _count_csv_data_rows,
    _is_bot_challenge_page,
    scrape_ufcstats_fights_to_csv,
)


class TestUfcstatsBotChallenge(unittest.TestCase):
    def test_detect_challenge_page(self) -> None:
        html = "<html><head><title>Loading…</title></head><body>Checking your browser</body></html>"
        soup = BeautifulSoup(html, "html.parser")
        self.assertTrue(_is_bot_challenge_page(soup))

    def test_keep_existing_csv_when_index_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ufcstats_fights.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "fight_id",
                        "fighter_a_id",
                        "fighter_b_id",
                        "winner_id",
                        "method",
                        "weight_class",
                        "date",
                    ],
                )
                w.writeheader()
                w.writerow(
                    {
                        "fight_id": "abc",
                        "fighter_a_id": "a",
                        "fighter_b_id": "b",
                        "winner_id": "a",
                        "method": "KO/TKO",
                        "weight_class": "Lightweight",
                        "date": "2020-01-01",
                    }
                )
            self.assertEqual(_count_csv_data_rows(out), 1)

            class FakeSession:
                def get(self, url, **kwargs):
                    class Resp:
                        text = "<html><title>Loading…</title><body>Checking your browser</body></html>"

                        def raise_for_status(self) -> None:
                            return None

                    return Resp()

            n = scrape_ufcstats_fights_to_csv(
                out,
                session=FakeSession(),  # type: ignore[arg-type]
            )
            self.assertEqual(n, 1)
            self.assertEqual(_count_csv_data_rows(out), 1)


if __name__ == "__main__":
    unittest.main()
