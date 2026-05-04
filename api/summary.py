"""Summarize a Binance announcement in Korean using Gemini Flash.

GET /api/summary?code=<article_code>
Response: { ok: true, title: "...", summary: "..." } or { ok: false, error: "..." }

Vercel Hobby plan caps total duration at 10s. Both upstream calls (Binance
detail + Gemini) must fit inside that, so we run a single attempt per call
with conservative timeouts. Retry is handled by user re-click on the UI side.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler

# --- Config ---
BINANCE_DETAIL_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
TIMEOUT_BINANCE_SEC = 3
TIMEOUT_GEMINI_SEC = 5
MAX_BODY_CHARS = 4000  # cap LLM input to keep latency/cost low

PROMPT_TEMPLATE = """\
다음은 Binance announcement 영어 본문이다. 한국어로 핵심만 3~5줄로 요약해라.

포함할 정보:
- 어떤 종류 이벤트 (Launchpool / Earn / 입금 보너스 / 신규 상장 / 거래 대회 등)
- 참여 조건 (자산, 락업 기간, 최소 금액 등)
- 보상 (APR, 보너스 금액, 토큰량 등)
- 마감일 또는 시작일

규칙:
- 사실만. 본문에 없는 정보는 추측 금지
- 사견 / 평가 금지
- 짧게

Title: {title}

Body:
{body}
"""


# --- HTTP helpers ---
def _json_request(req: urllib.request.Request, timeout: int, label: str) -> dict:
    """Single-attempt HTTP+JSON. Wraps errors with label + response body
    snippet so the frontend can identify which upstream failed.
    """
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            snippet = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            snippet = ""
        raise RuntimeError(f"{label}: HTTP {e.code} — {snippet}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"{label}: {e.__class__.__name__}: {e}") from e


# --- Binance detail ---
def fetch_binance_detail(code: str) -> dict:
    url = f"{BINANCE_DETAIL_URL}?code={urllib.parse.quote(code, safe='-_.~')}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; cex-event-feed/0.1)",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    })
    return _json_request(req, TIMEOUT_BINANCE_SEC, "binance detail")


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def extract_body(payload: dict) -> tuple[str, str]:
    """Return (title, body_text) from Binance detail payload.

    Body field name varies between endpoint versions — try common candidates.
    """
    data = payload.get("data") or {}
    title = data.get("title", "") or ""
    body = data.get("body") or data.get("content") or data.get("brief") or ""
    if isinstance(body, list):
        body = " ".join(str(b) for b in body if b)
    body = _HTML_TAG_RE.sub(" ", str(body))
    body = _WHITESPACE_RE.sub(" ", body).strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "..."
    return title, body


# --- Gemini ---
def call_gemini(prompt: str, api_key: str) -> str:
    url = f"{GEMINI_URL}?key={urllib.parse.quote(api_key, safe='')}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 600},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        },
    )
    resp = _json_request(req, TIMEOUT_GEMINI_SEC, "gemini")
    candidates = resp.get("candidates") or []
    if not candidates:
        raise RuntimeError("gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", []) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text.strip():
        raise RuntimeError("gemini returned empty text")
    return text.strip()


# --- Pipeline ---
def summarize(code: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")
    detail = fetch_binance_detail(code)
    title, body = extract_body(detail)
    if not body:
        raise RuntimeError("article body is empty — cannot summarize")
    prompt = PROMPT_TEMPLATE.format(title=title, body=body)
    summary = call_gemini(prompt, api_key)
    return {"title": title, "summary": summary}


# --- Vercel handler ---
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            code = (params.get("code") or [""])[0].strip()
            if not code:
                raise RuntimeError("missing 'code' query param")
            result = summarize(code)
            body = json.dumps({"ok": True, **result}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # pragma: no cover — top-level guard
            err = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err)
