"""
Tests for UFCStats **upcoming** event bout rows parsed locally (no HTTP).

WHAT RUNS
    Executes :func:`~src.data.ufcstats_upcoming.parse_upcoming_bouts_from_event_soup` on
    static HTML fragments shaped like UFCStats ``event-details`` bout tables.

HOW TO RUN (repo root = directory containing ``src/`` and ``tests/``)
    ``python -m unittest tests.test_upcoming_bouts_parse -v``

    More context on failure (Python 3.12+):
    ``python -m unittest tests.test_upcoming_bouts_parse -v --locals``

    Or:
    ``python tests/test_upcoming_bouts_parse.py``

READING FAILURES
    Assertions use ``subTest`` so you see which **field** failed (e.g. ``[fight_id]``).
    If the suite returns **no rows** for the good fixture, UFCStats likely changed
    column indices, classes, or ``data-link`` format — compare to
    ``src.data.ufcstats_upcoming._parse_bout_row``.
"""

from __future__ import annotations

import sys
import unittest

from bs4 import BeautifulSoup

from src.data.ufcstats_upcoming import parse_upcoming_bouts_from_event_soup


# Minimal clone of UFCStats markup: upcoming cards keep this ``tr.js-fight-details-click`` shape.
MINIMAL_EVENT_TABLE_SNIPPET = """
<table class="b-fight-details__table">
  <tbody>
    <tr class="b-fight-details__table-row js-fight-details-click" data-link="http://ufcstats.com/fight-details/e4aa608124896794">
      <td></td>
      <td>
        <p><a href="/fighter-details/040a74bb0a465c54" class="b-link">Arnold Allen</a></p>
        <p><a href="/fighter-details/20bccc9bb4ceb23e" class="b-link">Melquizael Costa</a></p>
      </td>
      <td></td><td></td><td></td><td></td>
      <td>Featherweight</td>
      <td></td><td></td><td></td>
    </tr>
  </tbody>
</table>
"""


def setUpModule() -> None:
    print(
        "\n"
        "================================================================================\n"
        " MODULE: tests.test_upcoming_bouts_parse\n"
        " Target: parse_upcoming_bouts_from_event_soup (offline; no UFCStats HTTP)\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


class TestUpcomingBoutParse(unittest.TestCase):
    """Valid vs invalid upcoming bout rows."""

    def test_01_fixture_row_yields_expected_bout_dictionary(self) -> None:
        """Fixture: one valid row -> one dict (fight + fighter IDs + Featherweight).

        Validates that UFCStats ordering (names in col 1, division text in col 6) still matches
        :func:`~src.data.ufcstats_upcoming._parse_bout_row` expectations.
        """
        soup = BeautifulSoup(MINIMAL_EVENT_TABLE_SNIPPET, "html.parser")
        print("    -- Parsing MINIMAL_EVENT_TABLE_SNIPPET ...", flush=True, file=sys.stderr)
        rows = parse_upcoming_bouts_from_event_soup(soup)
        print(f"    -- Parsed {len(rows)} bout row(s).", flush=True, file=sys.stderr)

        self.assertEqual(
            len(rows),
            1,
            msg="Expected exactly one bout; if 0 — selectors/HTML layout drift. If >1 — fixture grew extra rows.",
        )
        bout = rows[0]

        checks: dict[str, tuple[object, object]] = {
            "fight_id": ("e4aa608124896794", bout["fight_id"]),
            "fighter_a_id": ("040a74bb0a465c54", bout["fighter_a_id"]),
            "fighter_b_id": ("20bccc9bb4ceb23e", bout["fighter_b_id"]),
            "fighter_a_name": ("Arnold Allen", bout["fighter_a_name"]),
            "fighter_b_name": ("Melquizael Costa", bout["fighter_b_name"]),
            "weight_class": ("featherweight", bout["weight_class"]),
            "bout_order": (0, bout["bout_order"]),
        }
        for name, (want, got) in checks.items():
            with self.subTest(field=name):
                self.assertEqual(
                    want,
                    got,
                    msg=(
                        f"Field `{name}` mismatch — website export assumes corner order A,B from table rows."
                        f"\n    expected: {want!r}"
                        f"\n    actual:   {got!r}"
                    ),
                )

        with self.subTest(field="fight_url"):
            fu = bout.get("fight_url", "") or ""
            self.assertIn(
                "e4aa608124896794",
                fu,
                msg=f"fight_url should contain bout hex id; got {fu!r}",
            )

        print("    -- test_01: all assertions passed.\n", flush=True, file=sys.stderr)

    def test_02_incomplete_names_cell_is_skipped(self) -> None:
        """Fixture: only one fighter link -> zero rows (no half-parsed matchups downstream)."""

        incomplete = """
        <table class="b-fight-details__table">
          <tbody>
            <tr class="b-fight-details__table-row js-fight-details-click"
                data-link="http://ufcstats.com/fight-details/aaaaaaaaaaaaaaaa">
              <td></td>
              <td>
                <p><a href="/fighter-details/1111111111111111" class="b-link">Only One</a></p>
              </td>
              <td></td><td></td><td></td><td></td>
              <td>Lightweight</td>
              <td></td><td></td><td></td>
            </tr>
          </tbody>
        </table>
        """

        soup = BeautifulSoup(incomplete, "html.parser")
        print("    -- Parsing incomplete single-fighter snippet ...", flush=True, file=sys.stderr)
        rows = parse_upcoming_bouts_from_event_soup(soup)
        print(f"    -- Parsed {len(rows)} bout row(s) (want 0).", flush=True, file=sys.stderr)
        self.assertEqual(
            len(rows),
            0,
            msg="Parser must drop rows missing a second fighter `a.b-link`, not invent a matchup.",
        )
        print("    -- test_02: passed.\n", flush=True, file=sys.stderr)


def tearDownModule() -> None:
    print(
        "================================================================================\n"
        " END tests.test_upcoming_bouts_parse\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
