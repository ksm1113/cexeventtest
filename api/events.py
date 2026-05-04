"""Binance announcement collector — rule-matched CEX event feed.

Vercel Python serverless handler. Standard library only.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler

# --- Config ---
BINANCE_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query"
)
CATALOG_ID = 48  # General feed — community-known unofficial endpoint.
PAGE_SIZE = 30
# Vercel Hobby plan caps function duration at 10s. Worst case must fit:
# TIMEOUT_SEC * MAX_RETRIES + sum(backoff) <= 9s.
TIMEOUT_SEC = 4
MAX_RETRIES = 2
BACKOFF_BASE_SEC = 1  # 1s between retries (single retry → total ~9s worst)

_INCLUDE_KEYWORDS = (
    "launchpool", "bnb vault", "simple earn", "locked product",
    "dual investment", "staking", "earn",
    "deposit", "bonus", "cashback", "lock-up", "lockup",
    "promotion", "reward pool", "yield",
)
_EXCLUDE_KEYWORDS = (
    "trading competition", "trading contest", "tournament",
    "trade and win", "trading rewards",
)
# Word-boundary regex prevents false positives (e.g. "earn" matching "learn").
INCLUDE_PATTERNS = tuple(
    (kw, re.compile(r"\b" + re.escape(kw) + r"\b")) for kw in _INCLUDE_KEYWORDS
)
EXCLUDE_PATTERNS = tuple(
    (kw, re.compile(r"\b" + re.escape(kw) + r"\b")) for kw in _EXCLUDE_KEYWORDS
)
APR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


# --- Fetch ---
def fetch_binance(catalog_id: int = CATALOG_ID, page_size: int = PAGE_SIZE) -> dict:
    url = f"{BINANCE_URL}?catalogId={catalog_id}&pageNo=1&pageSize={page_size}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; cex-event-feed/0.1)",
        "Accept": "application/json",
        # Force plain response — gzip would break .decode/.json.loads downstream.
        "Accept-Encoding": "identity",
    }
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE_SEC)
    raise RuntimeError(f"binance fetch failed after {MAX_RETRIES} retries: {last_err}")


# --- Parse ---
def extract_articles(payload: dict) -> list[dict]:
    """Flatten catalogs[].articles[] into single list with catalog name attached."""
    data = payload.get("data") or {}
    catalogs = data.get("catalogs") or []
    flat: list[dict] = []
    if isinstance(catalogs, list):
        for cat in catalogs:
            if not isinstance(cat, dict):
                continue
            cat_name = cat.get("catalogName", "")
            for art in cat.get("articles", []) or []:
                if isinstance(art, dict):
                    art = dict(art)
                    art["_catalogName"] = cat_name
                    flat.append(art)
    # Some shapes return data.articles directly
    if not flat:
        articles = data.get("articles") or []
        if isinstance(articles, list):
            flat = [a for a in articles if isinstance(a, dict)]
    return flat


# --- Rule match ---
def match_rules(title: str, body: str = "") -> dict | None:
    text = f"{title} {body}".lower()
    if any(p.search(text) for _, p in EXCLUDE_PATTERNS):
        return None
    matched = [kw for kw, p in INCLUDE_PATTERNS if p.search(text)]
    if not matched:
        return None
    apy_hint = None
    m = APR_RE.search(text)
    if m:
        try:
            apy_hint = str(Decimal(m.group(1)))
        except (InvalidOperation, ValueError):
            apy_hint = None
    return {"matched": matched, "apy_hint": apy_hint}


# --- Build event ---
def to_iso(release_date) -> str | None:
    # bool is a subclass of int in Python — exclude before the numeric branch.
    if isinstance(release_date, bool):
        return None
    if isinstance(release_date, (int, float)):
        try:
            return datetime.fromtimestamp(release_date / 1000, tz=timezone.utc).isoformat()
        except (OverflowError, ValueError, OSError):
            return None
    if isinstance(release_date, str):
        return release_date
    return None


def _build_url(article: dict) -> str:
    """Binance announcement URL pattern: /en/support/announcement/{slug}.
    Only build URL when 'code' (slug) is present; numeric articleId alone
    does not produce a valid URL on this path.
    """
    code = article.get("code")
    if not isinstance(code, str) or not code:
        return ""
    safe_code = urllib.parse.quote(code, safe="-_.~")
    return f"https://www.binance.com/en/support/announcement/{safe_code}"


def build_event(article: dict, match: dict) -> dict:
    return {
        "exchange": "binance",
        "category": article.get("_catalogName", ""),
        "title": article.get("title", ""),
        "url": _build_url(article),
        "published_at": to_iso(article.get("releaseDate")),
        "matched": match["matched"],
        "apy_hint": match["apy_hint"],
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def collect() -> tuple[list[dict], int]:
    """Return (matched_events, raw_article_count). raw_count helps debug
    whether 0 results means upstream gave nothing vs filter dropped all."""
    payload = fetch_binance()
    articles = extract_articles(payload)
    out: list[dict] = []
    for art in articles:
        title = art.get("title", "")
        body = art.get("brief") or art.get("body") or ""
        m = match_rules(title, body)
        if m is None:
            continue
        out.append(build_event(art, m))
    out.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return out, len(articles)


# --- Vercel handler ---
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            events, raw_count = collect()
            body = json.dumps({
                "ok": True,
                "count": len(events),
                "raw_count": raw_count,
                "events": events,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "public, max-age=60")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # pragma: no cover — top-level guard
            err = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err)
