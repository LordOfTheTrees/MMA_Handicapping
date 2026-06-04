"""
Conservative HTTP client for ESPN's undocumented MMA APIs.

All GETs are disk-cached under ``data/cache/espn/`` and spaced by
``ESPN_REQUEST_DELAY_SEC`` so we do not hammer endpoints.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Seconds between network requests (not cache hits).
ESPN_REQUEST_DELAY_SEC: float = 0.45

SITE_API = "https://site.api.espn.com"
WEB_API = "https://site.web.api.espn.com"
CORE_API = "https://sports.core.api.espn.com/v2/sports/mma"

DEFAULT_CACHE_ROOT = Path("data/cache/espn")


class ESPNClient:
    def __init__(
        self,
        *,
        cache_dir: Path = DEFAULT_CACHE_ROOT,
        request_delay_sec: float = ESPN_REQUEST_DELAY_SEC,
        log_network: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay_sec = request_delay_sec
        self.log_network = log_network
        self._last_request_at: float = 0.0
        self.network_requests: int = 0
        self.cache_hits: int = 0

    def _cache_path_for_url(self, url: str) -> Path:
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_") or "root"
        query = parsed.query.replace("/", "_")
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{parsed.netloc}_{path}_{query}")[:200]
        return self.cache_dir / f"{safe}.json"

    def get_json(self, url: str, *, use_cache: bool = True) -> Dict[str, Any]:
        cache_path = self._cache_path_for_url(url)
        if use_cache and cache_path.is_file():
            self.cache_hits += 1
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)

        self._throttle()
        self.network_requests += 1
        if self.log_network:
            short = url if len(url) <= 100 else url[:97] + "..."
            print(f"    [espn GET] {short}", flush=True)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "MMA-Handicapping/1.0 (research; cached incremental)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ESPN HTTP {e.code} for {url}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"ESPN request failed for {url}: {e}") from e

        if use_cache:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        return payload

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < self.request_delay_sec:
            time.sleep(self.request_delay_sec - elapsed)
        self._last_request_at = time.monotonic()

    def list_season_years(self) -> List[int]:
        url = f"{CORE_API}/leagues/ufc/seasons?limit=100"
        data = self.get_json(url)
        years: List[int] = []
        for item in data.get("items") or []:
            ref = item.get("$ref") or ""
            m = re.search(r"/seasons/(\d{4})\?", ref) or re.search(r"/seasons/(\d{4})$", ref)
            if m:
                years.append(int(m.group(1)))
        return sorted(set(years))

    def list_event_refs(self, season_year: int) -> List[str]:
        url = f"{CORE_API}/leagues/ufc/seasons/{season_year}/types/2/events?limit=200"
        data = self.get_json(url)
        return [item["$ref"] for item in (data.get("items") or []) if item.get("$ref")]

    def fetch_event(self, event_ref: str) -> Dict[str, Any]:
        return self.get_json(event_ref)

    def fetch_fightcenter(self, event_id: str) -> Dict[str, Any]:
        url = (
            f"{WEB_API}/apis/common/v3/sports/mma/ufc/fightcenter/{event_id}"
            "?region=us&lang=en&contentorigin=espn"
        )
        return self.get_json(url)

    def fetch_competition(self, event_id: str, competition_id: str) -> Dict[str, Any]:
        url = f"{CORE_API}/leagues/ufc/events/{event_id}/competitions/{competition_id}"
        return self.get_json(url)

    def fetch_competition_status(self, status_ref: str) -> Dict[str, Any]:
        return self.get_json(status_ref)

    def list_competitor_refs(self, event_id: str, competition_id: str) -> List[str]:
        url = (
            f"{CORE_API}/leagues/ufc/events/{event_id}/competitions/{competition_id}"
            "/competitors?limit=10"
        )
        data = self.get_json(url)
        return [item["$ref"] for item in (data.get("items") or []) if item.get("$ref")]

    def fetch_competitor(self, competitor_ref: str) -> Dict[str, Any]:
        return self.get_json(competitor_ref)

    def fetch_competitor_statistics(self, statistics_ref: str) -> Dict[str, Any]:
        return self.get_json(statistics_ref)

    def fetch_athlete(self, athlete_id: str) -> Dict[str, Any]:
        url = f"{CORE_API}/athletes/{athlete_id}"
        return self.get_json(url)

    def fetch_athlete_eventlog(self, athlete_id: str) -> Dict[str, Any]:
        # site.api …/eventlog returns a rankings shell (no fight items). Core v2 is correct.
        url = f"{CORE_API}/athletes/{athlete_id}/eventlog?lang=en&region=us"
        return self.get_json(url)

    def fetch_athlete_records(self, athlete_id: str) -> Dict[str, Any]:
        url = f"{CORE_API}/athletes/{athlete_id}/records?lang=en&region=us"
        return self.get_json(url)
