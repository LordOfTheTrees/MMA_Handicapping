"""
Scrape UFCStats.com into a fights CSV (columns expected by ``load_ufcstats_fights``).

HTTP uses ``curl_cffi`` with Chrome impersonation plus Referer chains.

**Rate limiting:** ``REQUEST_DELAY_SEC`` at module top is the delay between requests.
Override at runtime before scraping, or use ``--sleep`` on the CLI (sets the global).

**Default output:** ``ufcstats_fights.csv`` under ``--data-dir`` or ``./data``.
Legacy ``tier1_ufcstats.csv`` is still loaded by the pipeline if the new file is absent.

Usage (from repo root)::

    pip install -r requirements.txt
    python -m src.data.ufcstats_scraper --data-dir ./data
    python -m src.data.ufcstats_scraper --out ./out.csv --max-events 3 --max-fights 20
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests

from src.data.loader import METHOD_MAP, WEIGHT_CLASS_MAP

# --- Runtime-tunable request spacing (seconds). CLI ``--sleep`` assigns this in ``main``. ---
REQUEST_DELAY_SEC: float = 0.2

# Default basename next to ``--data-dir`` / ``./data``.
DEFAULT_UFCSTATS_FIGHTS_CSV = "ufcstats_fights.csv"

BASE = "http://ufcstats.com"
COMPLETED_EVENTS_URL = f"{BASE}/statistics/events/completed?page=all"
BROWSER_IMPERSONATE = "chrome131"


def _throttle(override_sec: Optional[float] = None) -> None:
    """Sleep between requests. ``override_sec`` (e.g. from inventory tools) skips the global."""
    time.sleep(REQUEST_DELAY_SEC if override_sec is None else override_sec)


@dataclass
class FighterFightRow:
    fighter_id: str
    sig_landed: Optional[int]
    sig_attempted: Optional[int]
    td_landed: Optional[int]
    td_attempted: Optional[int]
    ctrl_sec: Optional[int]
    sub_attempts: Optional[int]


@dataclass(frozen=True)
class ExpectedFight:
    fight_id: str
    fight_url: str
    event_url: str
    event_date: date


def _session() -> requests.Session:
    return requests.Session()


def fetch_soup(
    session: requests.Session,
    url: str,
    *,
    referer: Optional[str] = None,
) -> BeautifulSoup:
    kwargs = {
        "timeout": 60,
        "impersonate": BROWSER_IMPERSONATE,
        "allow_redirects": True,
    }
    if referer:
        kwargs["referer"] = referer
    r = session.get(url, **kwargs)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def fighter_id_from_href(href: str) -> Optional[str]:
    if not href:
        return None
    path = urlparse(href.strip()).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "fighter-details":
        return parts[1]
    return None


def parse_event_date(soup: BeautifulSoup) -> Optional[date]:
    for li in soup.select("li.b-list__box-list-item"):
        title = li.find("i", class_="b-list__box-item-title")
        if not title:
            continue
        if "Date:" not in title.get_text():
            continue
        rest = li.get_text().replace(title.get_text(), "", 1).strip()
        try:
            return datetime.strptime(rest, "%B %d, %Y").date()
        except ValueError:
            return None
    return None


def _canonical_ufcstats_http_url(href: str) -> str:
    full = href if href.startswith("http") else f"{BASE}{href if href.startswith('/') else '/' + href}"
    p = urlparse(full)
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "ufcstats.com":
        return full
    path = p.path or ""
    query = f"?{p.query}" if p.query else ""
    return f"http://ufcstats.com{path}{query}"


def iter_completed_event_urls(soup: BeautifulSoup) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()
    for a in soup.select('a[href*="event-details"]'):
        href = a.get("href")
        if not href or "event-details" not in href:
            continue
        full = _canonical_ufcstats_http_url(href)
        if full not in seen:
            seen.add(full)
            urls.append(full)
    return urls


def fight_urls_from_event_page(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for tr in soup.select("tr.js-fight-details-click"):
        link = tr.get("data-link") or ""
        m = re.search(r"fight-details/([a-f0-9]+)", link)
        if not m:
            continue
        fid = m.group(1)
        url = f"{BASE}/fight-details/{fid}"
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _parse_of_count(cell_text: str) -> Tuple[Optional[int], Optional[int]]:
    t = cell_text.strip().replace("\xa0", " ")
    m = re.search(r"(\d+)\s+of\s+(\d+)", t)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_ctrl_seconds(cell_text: str) -> Optional[int]:
    t = cell_text.strip()
    if t in ("---", "—", "-"):
        return None
    parts = t.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def _parse_int_cell(cell_text: str) -> Optional[int]:
    t = cell_text.strip()
    if t in ("---", "—", "-", ""):
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _find_totals_table(soup: BeautifulSoup):
    for table in soup.find_all("table"):
        if "js-fight-table" in (table.get("class") or []):
            continue
        thead = table.find("thead")
        if thead and "Sig. str." in thead.get_text():
            return table
    return None


def _parse_fight_meta(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    fight_block = soup.select_one(".b-fight-details__fight")
    if not fight_block:
        return None, None, None, None

    method_raw: Optional[str] = None
    round_n: Optional[int] = None
    clock_sec: Optional[int] = None
    round_length_sec: Optional[int] = None

    for para in fight_block.select("p.b-fight-details__text"):
        for item in para.select(
            ".b-fight-details__text-item, .b-fight-details__text-item_first"
        ):
            label = item.find("i", class_="b-fight-details__label")
            if not label:
                continue
            key = label.get_text(strip=True).lower().rstrip(":")
            tail = ""
            for child in item.children:
                if child is label:
                    continue
                if hasattr(child, "get_text"):
                    tail += child.get_text()
                else:
                    tail += str(child)
            tail = tail.strip()

            if key == "method":
                inner = item.find("i", style=re.compile(r"font-style:\s*normal", re.I))
                method_raw = (inner or item).get_text(strip=True)
            elif key == "round":
                try:
                    round_n = int(tail)
                except ValueError:
                    round_n = None
            elif key == "time":
                m = re.match(r"(\d+):(\d+)$", tail.strip())
                if m:
                    clock_sec = int(m.group(1)) * 60 + int(m.group(2))
            elif key == "time format":
                rm = re.search(r"\(([\d\-]+)\)", tail)
                if rm:
                    first = rm.group(1).split("-")[0]
                    try:
                        round_length_sec = int(first) * 60
                    except ValueError:
                        round_length_sec = 5 * 60

    return method_raw, round_n, clock_sec, round_length_sec


def _normalize_method(method_raw: Optional[str]) -> Optional[str]:
    if not method_raw:
        return None
    s = method_raw.strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in ("ko/tko", "ko", "tko"):
        return "ko/tko"
    if "ko/tko" in s or s == "ko" or s == "tko":
        return "ko/tko"
    # Doctor's stoppage / punch TKOs: credit as finish for winner (they caused the stoppage).
    if re.match(r"^(tko|ko)\b", s):
        return "ko/tko"
    if "submission" in s or s == "sub":
        return "submission"
    if "draw" in s:
        return "draw"
    if "unanimous" in s:
        return "unanimous decision"
    if "split" in s and "decision" in s:
        return "split decision"
    if "majority" in s:
        return "majority decision"
    if "no contest" in s or s == "nc" or "could not continue" in s:
        return "no contest"
    if "disqual" in s or s == "dq":
        return "dq"
    if s in METHOD_MAP:
        return s
    if s in ("u-dec", "u-dec.", "unanimous dec", "unanimous dec."):
        return "unanimous decision"
    if s in ("s-dec", "s-dec.", "split dec", "split dec."):
        return "split decision"
    if s in ("m-dec", "m-dec.", "majority dec", "majority dec."):
        return "majority decision"
    return None


def _normalize_title_text(title_text: str) -> str:
    return re.sub(r"\s+", " ", (title_text or "")).strip()


def _canonical_weight_class_from_title(title_text: str) -> Optional[str]:
    """
    Return loader key (e.g. ``lightweight``) or ``None`` if the title is non-standard.

    Tournament / long titles embed the division as a substring (e.g. "... Lightweight Tournament ...");
    we match the longest ``WEIGHT_CLASS_MAP`` key first so ``light heavyweight`` wins over
    ``lightweight``. Catch-weight bouts use the fixed label ``catch_weight`` (numeric weights
    are not on the fight page). Other odd titles fall through to ``None`` for raw CSV fallback.
    """
    t = _normalize_title_text(title_text)
    if not t:
        return None
    t = re.sub(r"^UFC\s+", "", t, flags=re.I).strip()
    for suffix in (
        "Interim Title Bout",
        "Title Bout",
        "Championship Bout",
        "Tournament Bout",
        "Bout",
    ):
        if t.endswith(suffix):
            t = t[: -len(suffix)].strip()
    t = re.sub(r"^interim\s+", "", t, flags=re.I).strip()
    wc = t.lower().strip()
    if not wc:
        return None
    if wc in WEIGHT_CLASS_MAP:
        return wc

    haystack = _normalize_title_text(title_text).lower()
    if "catch weight" in haystack or "catch-weight" in haystack:
        return "catch_weight"

    for key in sorted(WEIGHT_CLASS_MAP.keys(), key=len, reverse=True):
        if key in haystack:
            return key
    return None


def _parse_weight_class(soup: BeautifulSoup) -> Optional[str]:
    """
    Canonical division string for ``WEIGHT_CLASS_MAP``, or normalized page title
    (lowercased) when non-standard so the CSV preserves UFCStats wording.
    """
    title_el = soup.select_one(".b-fight-details__fight-title")
    if not title_el:
        return None
    raw = _normalize_title_text(title_el.get_text())
    canon = _canonical_weight_class_from_title(raw)
    if canon:
        return canon
    return raw.lower() if raw else None


def _person_rows(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for person in soup.select(".b-fight-details__person"):
        st = person.select_one(".b-fight-details__person-status")
        link = person.select_one("a.b-fight-details__person-link")
        if not st or not link:
            continue
        fid = fighter_id_from_href(link.get("href", ""))
        if not fid:
            continue
        flag = st.get_text(strip=True).upper()
        rows.append((fid, flag))
    return rows


def _totals_fighter_rows(table) -> Optional[List[FighterFightRow]]:
    tbody = table.find("tbody")
    if not tbody:
        return None
    tr = tbody.find("tr")
    if not tr:
        return None
    tds = tr.find_all("td", recursive=False)
    if len(tds) < 10:
        return None

    names_td = tds[0]
    links = names_td.select("p.b-fight-details__table-text a.b-link")
    if len(links) < 2:
        return None

    def col_vals(idx: int) -> List[str]:
        td = tds[idx]
        ps = td.select("p.b-fight-details__table-text")
        return [p.get_text(strip=True) for p in ps]

    kd = col_vals(1)
    sig = col_vals(2)
    td_ = col_vals(5)
    sub = col_vals(7)
    ctrl = col_vals(9)

    out: List[FighterFightRow] = []
    for i in range(2):
        landed, attempted = _parse_of_count(sig[i] if i < len(sig) else "")
        td_l, td_a = _parse_of_count(td_[i] if i < len(td_) else "")
        out.append(
            FighterFightRow(
                fighter_id=fighter_id_from_href(links[i].get("href", "")) or "",
                sig_landed=landed,
                sig_attempted=attempted,
                td_landed=td_l,
                td_attempted=td_a,
                ctrl_sec=_parse_ctrl_seconds(ctrl[i] if i < len(ctrl) else ""),
                sub_attempts=_parse_int_cell(sub[i] if i < len(sub) else ""),
            )
        )
    return out


def _winner_id_from_flags(person_rows: List[Tuple[str, str]], method_norm: Optional[str]) -> Optional[str]:
    if method_norm in ("draw", "no contest"):
        return None
    winners = [fid for fid, fl in person_rows if fl == "W"]
    if len(winners) == 1:
        return winners[0]
    if len(winners) == 0 and method_norm == "dq":
        losers = [fid for fid, fl in person_rows if fl == "L"]
        if len(losers) == 1:
            other = [fid for fid, _ in person_rows if fid != losers[0]]
            if len(other) == 1:
                return other[0]
    return None


def _fight_duration_sec(
    round_n: Optional[int],
    clock_sec: Optional[int],
    round_length_sec: Optional[int],
) -> Optional[int]:
    if round_n is None or clock_sec is None:
        return None
    rlen = round_length_sec or 300
    return (round_n - 1) * rlen + clock_sec


def parse_fight_page(
    soup: BeautifulSoup,
    fight_id: str,
    event_date: date,
) -> Optional[Dict[str, Any]]:
    method_raw, round_n, clock_sec, round_len = _parse_fight_meta(soup)
    method_norm = _normalize_method(method_raw)
    persons = _person_rows(soup)
    if len(persons) == 2 and all(fl.strip().upper() == "D" for _, fl in persons):
        method_norm = "draw"
    if len(persons) == 2 and all(fl.strip().upper() == "NC" for _, fl in persons):
        method_norm = "no contest"

    wc = _parse_weight_class(soup)
    if method_norm is None or wc is None:
        return None

    if len(persons) != 2:
        return None

    table = _find_totals_table(soup)
    if not table:
        return None
    stats_rows = _totals_fighter_rows(table)
    if not stats_rows or len(stats_rows) != 2:
        return None

    by_id = {r.fighter_id: r for r in stats_rows}
    page_order_ids = [persons[0][0], persons[1][0]]
    if set(page_order_ids) != set(by_id.keys()):
        return None

    winner_id = _winner_id_from_flags(persons, method_norm)

    r0 = by_id[page_order_ids[0]]
    r1 = by_id[page_order_ids[1]]

    id_a, id_b = sorted(page_order_ids)
    if id_a == page_order_ids[0]:
        sa, sb = r0, r1
    else:
        sa, sb = r1, r0

    fight_time = _fight_duration_sec(round_n, clock_sec, round_len)

    row = {
        "fight_id": fight_id,
        "fighter_a_id": id_a,
        "fighter_b_id": id_b,
        "winner_id": winner_id or "",
        "method": method_norm,
        "weight_class": wc,
        "date": event_date.isoformat(),
        "fight_time_sec": fight_time if fight_time is not None else "",
        "a_sig_str_landed": sa.sig_landed if sa.sig_landed is not None else "",
        "a_sig_str_attempted": sa.sig_attempted if sa.sig_attempted is not None else "",
        "a_sig_str_absorbed": sb.sig_landed if sb.sig_landed is not None else "",
        "a_td_landed": sa.td_landed if sa.td_landed is not None else "",
        "a_td_attempted": sa.td_attempted if sa.td_attempted is not None else "",
        "a_ctrl_time_sec": sa.ctrl_sec if sa.ctrl_sec is not None else "",
        "a_sub_attempts": sa.sub_attempts if sa.sub_attempts is not None else "",
        "b_sig_str_landed": sb.sig_landed if sb.sig_landed is not None else "",
        "b_sig_str_attempted": sb.sig_attempted if sb.sig_attempted is not None else "",
        "b_sig_str_absorbed": sa.sig_landed if sa.sig_landed is not None else "",
        "b_td_landed": sb.td_landed if sb.td_landed is not None else "",
        "b_td_attempted": sb.td_attempted if sb.td_attempted is not None else "",
        "b_ctrl_time_sec": sb.ctrl_sec if sb.ctrl_sec is not None else "",
        "b_sub_attempts": sb.sub_attempts if sb.sub_attempts is not None else "",
    }
    if winner_id and winner_id not in (id_a, id_b):
        return None
    return row


def diagnose_fight_parse_failure(
    soup: BeautifulSoup,
    fight_id: str,
    event_date: date,
) -> str:
    method_raw, _round_n, _clock_sec, _round_len = _parse_fight_meta(soup)
    method_norm = _normalize_method(method_raw)
    persons_pre = _person_rows(soup)
    if len(persons_pre) == 2 and all(fl.strip().upper() == "D" for _, fl in persons_pre):
        method_norm = "draw"
    if len(persons_pre) == 2 and all(fl.strip().upper() == "NC" for _, fl in persons_pre):
        method_norm = "no contest"

    if method_norm is None:
        return f"unmapped_method:{method_raw!r}"

    wc = _parse_weight_class(soup)
    if wc is None:
        title_el = soup.select_one(".b-fight-details__fight-title")
        raw_title = re.sub(r"\s+", " ", (title_el.get_text() if title_el else "") or "").strip()
        return f"unmapped_weight_class:{raw_title!r}"

    persons = persons_pre
    if len(persons) != 2:
        flags = [fl for _, fl in persons]
        return f"person_count:{len(persons)} flags={flags!r}"

    table = _find_totals_table(soup)
    if not table:
        return "no_totals_table"

    stats_rows = _totals_fighter_rows(table)
    if not stats_rows or len(stats_rows) != 2:
        return "totals_row_parse_failed"

    by_id = {r.fighter_id: r for r in stats_rows}
    page_order_ids = [persons[0][0], persons[1][0]]
    if set(page_order_ids) != set(by_id.keys()):
        return (
            f"banner_vs_table_id_mismatch "
            f"banner={page_order_ids} table={sorted(by_id.keys())}"
        )

    id_a, id_b = sorted(page_order_ids)
    winner_id = _winner_id_from_flags(persons, method_norm)
    if winner_id and winner_id not in (id_a, id_b):
        return f"winner_id_not_fighter:{winner_id!r}"

    if parse_fight_page(soup, fight_id, event_date) is None:
        return "parse_failed_unknown"
    return "ok"


def iter_expected_fights_from_completed_events(
    *,
    max_events: Optional[int] = None,
    session: Optional[requests.Session] = None,
    request_delay_sec: Optional[float] = None,
) -> Iterator[ExpectedFight]:
    sess = session or _session()
    index_soup = fetch_soup(sess, COMPLETED_EVENTS_URL, referer=f"{BASE}/statistics/events/")
    event_urls = iter_completed_event_urls(index_soup)
    if max_events is not None:
        event_urls = event_urls[:max_events]

    for event_url in event_urls:
        _throttle(request_delay_sec)
        try:
            ev_soup = fetch_soup(sess, event_url, referer=COMPLETED_EVENTS_URL)
        except requests.RequestsError:
            continue

        ev_date = parse_event_date(ev_soup)
        # Only past events: same-day cards often have no results on fight pages yet.
        if ev_date is None or ev_date >= date.today():
            continue

        for furl in fight_urls_from_event_page(ev_soup):
            m = re.search(r"fight-details/([a-f0-9]+)", furl)
            if not m:
                continue
            fid = m.group(1)
            full_url = furl if furl.startswith("http") else f"{BASE}/{furl.lstrip('/')}"
            yield ExpectedFight(
                fight_id=fid,
                fight_url=full_url,
                event_url=event_url,
                event_date=ev_date,
            )


CSV_FIELDS = [
    "fight_id",
    "fighter_a_id",
    "fighter_b_id",
    "winner_id",
    "method",
    "weight_class",
    "date",
    "fight_time_sec",
    "a_sig_str_landed",
    "a_sig_str_attempted",
    "a_sig_str_absorbed",
    "a_td_landed",
    "a_td_attempted",
    "a_ctrl_time_sec",
    "a_sub_attempts",
    "b_sig_str_landed",
    "b_sig_str_attempted",
    "b_sig_str_absorbed",
    "b_td_landed",
    "b_td_attempted",
    "b_ctrl_time_sec",
    "b_sub_attempts",
]

FAILED_ENTRY_FIELDS = [
    "fight_id",
    "fight_url",
    "event_url",
    "event_date",
    "failure_kind",
    "detail",
]


def _append_failed_entry(
    failed: List[Dict[str, str]],
    *,
    fight_id: str,
    fight_url: str,
    event_url: str,
    event_date: Optional[date],
    failure_kind: str,
    detail: str,
) -> None:
    failed.append(
        {
            "fight_id": fight_id,
            "fight_url": fight_url,
            "event_url": event_url,
            "event_date": event_date.isoformat() if event_date else "",
            "failure_kind": failure_kind,
            "detail": detail,
        }
    )
    print(
        f"    [failed {failure_kind}] fight_id={fight_id or '(none)'} | {detail}",
        flush=True,
    )


def scrape_ufcstats_fights_to_csv(
    out_path: Path,
    *,
    max_events: Optional[int] = None,
    max_fights: Optional[int] = None,
    session: Optional[requests.Session] = None,
    failed_entries_path: Optional[Path] = None,
) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if failed_entries_path is None:
        failed_entries_path = out_path.parent / "failed_entries.csv"
    else:
        failed_entries_path = Path(failed_entries_path)
    failed_entries_path.parent.mkdir(parents=True, exist_ok=True)
    sess = session or _session()

    print("Fetching event index ...", flush=True)
    index_soup = fetch_soup(sess, COMPLETED_EVENTS_URL, referer=f"{BASE}/statistics/events/")
    event_urls = iter_completed_event_urls(index_soup)
    if max_events is not None:
        event_urls = event_urls[:max_events]

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    n_fights = 0
    n_skipped = 0

    for ei, event_url in enumerate(event_urls):
        _throttle()
        try:
            ev_soup = fetch_soup(sess, event_url, referer=COMPLETED_EVENTS_URL)
        except requests.RequestsError as e:
            print(f"  [skip event] {event_url}: {e}", flush=True)
            continue

        ev_date = parse_event_date(ev_soup)
        if ev_date is None:
            print(f"  [skip event] no date: {event_url}", flush=True)
            continue
        if ev_date >= date.today():
            print(
                f"  [skip event] today/future card (no full results yet) {ev_date}: {event_url}",
                flush=True,
            )
            continue

        furls = fight_urls_from_event_page(ev_soup)
        print(f"  Event {ei + 1}/{len(event_urls)} ({ev_date}): {len(furls)} fights", flush=True)

        for furl in furls:
            if max_fights is not None and n_fights >= max_fights:
                break
            full_fight_url = furl if furl.startswith("http") else f"{BASE}/{furl.lstrip('/')}"
            m = re.search(r"fight-details/([a-f0-9]+)", furl)
            if not m:
                n_skipped += 1
                _append_failed_entry(
                    failed,
                    fight_id="",
                    fight_url=full_fight_url,
                    event_url=event_url,
                    event_date=ev_date,
                    failure_kind="bad_fight_url",
                    detail="no fight-details id in data-link",
                )
                continue
            fid = m.group(1)
            _throttle()
            try:
                fsoup = fetch_soup(sess, full_fight_url, referer=event_url)
            except requests.RequestsError as e:
                n_skipped += 1
                _append_failed_entry(
                    failed,
                    fight_id=fid,
                    fight_url=full_fight_url,
                    event_url=event_url,
                    event_date=ev_date,
                    failure_kind="fetch_error",
                    detail=str(e),
                )
                continue

            parsed = parse_fight_page(fsoup, fid, ev_date)
            if not parsed:
                n_skipped += 1
                reason = diagnose_fight_parse_failure(fsoup, fid, ev_date)
                _append_failed_entry(
                    failed,
                    fight_id=fid,
                    fight_url=full_fight_url,
                    event_url=event_url,
                    event_date=ev_date,
                    failure_kind="parse_error",
                    detail=reason,
                )
                continue
            rows.append(parsed)
            n_fights += 1

        if max_fights is not None and n_fights >= max_fights:
            break

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open(failed_entries_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FAILED_ENTRY_FIELDS)
        w.writeheader()
        w.writerows(failed)

    print(f"Wrote {len(rows)} rows -> {out_path} (skipped/problem fights: {n_skipped})", flush=True)
    print(f"Wrote {len(failed)} failed entries -> {failed_entries_path}", flush=True)
    return len(rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Scrape UFCStats into fights CSV")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Write <data-dir>/{DEFAULT_UFCSTATS_FIGHTS_CSV}",
    )
    p.add_argument("--sleep", type=float, default=0.2, help="Sets REQUEST_DELAY_SEC (seconds)")
    p.add_argument(
        "--failed-entries",
        type=Path,
        default=None,
        help="CSV for fetch/parse failures (default: <out-dir>/failed_entries.csv)",
    )
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--max-fights", type=int, default=None)
    args = p.parse_args(argv)

    global REQUEST_DELAY_SEC
    REQUEST_DELAY_SEC = args.sleep

    if args.data_dir:
        out = Path(args.data_dir) / DEFAULT_UFCSTATS_FIGHTS_CSV
    elif args.out:
        out = args.out
    else:
        out = Path("data") / DEFAULT_UFCSTATS_FIGHTS_CSV

    scrape_ufcstats_fights_to_csv(
        out,
        max_events=args.max_events,
        max_fights=args.max_fights,
        failed_entries_path=args.failed_entries,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
