"""
Upcoming UFCStats event cards — listings only, **not** merged into training CSVs.

Completed fight scraping intentionally skips today/future dates (ADR-05). This module
fetches http://ufcstats.com/statistics/events/upcoming and per-event bout rows so the
site / export pipeline can ship ``upcoming_cards.json`` without polluting
``ufcstats_fights.csv``.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests

from .loader import _coerce_weight_class_from_cell
from .schema import WeightClass
from .ufcstats_scraper import (
    BASE,
    fetch_soup,
    fighter_id_from_href,
    iter_completed_event_urls,
    parse_event_date,
    _session,
    _throttle,
)

UPCOMING_EVENTS_URL = f"{BASE}/statistics/events/upcoming"
DEFAULT_UPCOMING_CARDS_JSON = "upcoming_cards.json"


def _event_slug_from_url(event_url: str) -> str:
    path = urlparse(event_url).path.strip("/").split("/")
    return path[-1] if path else event_url


def _event_title_from_soup(soup: BeautifulSoup) -> str:
    h2 = soup.select_one("h2.b-content__title span")
    if h2:
        return h2.get_text(strip=True)
    h2b = soup.select_one("h2.b-content__title")
    if h2b:
        return h2b.get_text(strip=True)
    h2c = soup.select_one("h2")
    return h2c.get_text(strip=True) if h2c else ""


def _parse_location(soup: BeautifulSoup) -> Optional[str]:
    for li in soup.select("li.b-list__box-list-item"):
        title = li.find("i", class_="b-list__box-item-title")
        if not title:
            continue
        if "Location:" not in title.get_text():
            continue
        rest = li.get_text().replace(title.get_text(), "", 1).strip()
        return rest or None
    return None


def _parse_bout_row(tr: Any, bout_order: int) -> Optional[Dict[str, Any]]:
    """One ``tr.b-fight-details__table-row.js-fight-details-click`` from an event page."""
    link = tr.get("data-link") or ""
    m = re.search(r"fight-details/([a-f0-9]+)", link)
    if not m:
        return None
    fight_id = m.group(1)
    fight_url = link if link.startswith("http") else f"{BASE}/{link.lstrip('/')}"

    cells = tr.find_all("td", recursive=False)
    if len(cells) < 7:
        return None

    names_td = cells[1]
    flinks = names_td.select("a.b-link")
    if len(flinks) < 2:
        return None
    id_a = fighter_id_from_href(flinks[0].get("href", ""))
    id_b = fighter_id_from_href(flinks[1].get("href", ""))
    if not id_a or not id_b:
        return None
    name_a = flinks[0].get_text(strip=True)
    name_b = flinks[1].get_text(strip=True)

    wc_cell = cells[6].get_text(strip=True)
    wc_enum, wc_raw = _coerce_weight_class_from_cell(wc_cell)

    wc_value = wc_enum.value if wc_enum is not None else WeightClass.UNKNOWN.value

    return {
        "bout_order": bout_order,
        "fight_id": fight_id,
        "fight_url": fight_url,
        "fighter_a_id": id_a,
        "fighter_b_id": id_b,
        "fighter_a_name": name_a,
        "fighter_b_name": name_b,
        "weight_class": wc_value,
        "weight_class_raw": wc_raw or wc_cell,
    }


def parse_upcoming_bouts_from_event_soup(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract scheduled bouts from an event-details page (works for future cards)."""
    rows = soup.select("tr.b-fight-details__table-row.js-fight-details-click")
    out: List[Dict[str, Any]] = []
    for i, tr in enumerate(rows):
        row = _parse_bout_row(tr, bout_order=i)
        if row:
            out.append(row)
    return out


def scrape_upcoming_cards(
    *,
    session: Optional[requests.Session] = None,
    max_events: Optional[int] = None,
    request_delay_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return a JSON-serializable document of upcoming events and bouts.

    Does not write the training fights CSV and does not call ``parse_fight_page``.
    """
    sess = session or _session()
    index_soup = fetch_soup(sess, UPCOMING_EVENTS_URL, referer=f"{BASE}/statistics/events/")
    event_urls = iter_completed_event_urls(index_soup)
    if max_events is not None:
        event_urls = event_urls[: max_events]

    events_out: List[Dict[str, Any]] = []
    for event_url in event_urls:
        _throttle(request_delay_sec)
        try:
            ev_soup = fetch_soup(sess, event_url, referer=UPCOMING_EVENTS_URL)
        except requests.RequestsError:
            continue

        ev_date = parse_event_date(ev_soup)
        if ev_date is None:
            continue
        # Upcoming listing should not include past dates; skip if stale.
        if ev_date < date.today():
            continue

        title = _event_title_from_soup(ev_soup)
        location = _parse_location(ev_soup)
        bouts = parse_upcoming_bouts_from_event_soup(ev_soup)
        events_out.append(
            {
                "event_url": event_url,
                "event_id": _event_slug_from_url(event_url),
                "event_title": title,
                "event_date": ev_date.isoformat(),
                "location": location,
                "bouts": bouts,
            }
        )

    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "source": UPCOMING_EVENTS_URL,
        "scraped_at": scraped_at,
        "events": events_out,
    }


def scrape_upcoming_cards_to_path(
    path: Path,
    *,
    max_events: Optional[int] = None,
    request_delay_sec: Optional[float] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = scrape_upcoming_cards(max_events=max_events, request_delay_sec=request_delay_sec)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    import src.data.ufcstats_scraper as ufc_sc

    p = argparse.ArgumentParser(description="Scrape UFCStats upcoming events into JSON (not for training).")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for upcoming_cards.json")
    p.add_argument("--out", type=Path, default=None, help="Output file (default: data-dir/upcoming_cards.json)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--sleep", type=float, default=None, help="Override REQUEST_DELAY_SEC between requests")
    args = p.parse_args(argv)

    if args.sleep is not None:
        ufc_sc.REQUEST_DELAY_SEC = args.sleep

    out = args.out or (Path(args.data_dir) / DEFAULT_UPCOMING_CARDS_JSON)
    scrape_upcoming_cards_to_path(out, max_events=args.max_events, request_delay_sec=args.sleep)
    print(f"Wrote {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
