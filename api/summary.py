"""Summarize a Binance announcement in Korean using Gemini Flash.

GET /api/summary?code=<article_code>
Response: { ok: true, title: "...", summary: "..." } or { ok: false, error: "..." }

The bapi detail endpoint is not part of any public API surface (the
announcement page is server-rendered, so the browser never makes a detail
XHR). Instead we fetch the SSR HTML page and extract the article body —
preferring Next.js __NEXT_DATA__ JSON when present, falling back to
stripping all tags.

Vercel Hobby plan caps total duration at 10s, so each upstream call runs
once with a conservative timeout. User re-clicks for retries.
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
BINANCE_PAGE_URL = "https://www.binance.com/en/support/announcement/{code}"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
TIMEOUT_HTML_SEC = 5
TIMEOUT_GEMINI_SEC = 4
MAX_BODY_CHARS = 6000  # cap LLM input

PROMPT_TEMPLATE = """\
다음은 Binance announcement 페이지에서 추출한 영어 본문이다.
페이지 navigation/footer 같은 noise가 섞여 있을 수 있으니, announcement 핵심 정보만 골라 한국어로 3~5줄 요약해라.

포함할 정보:
- 어떤 종류 이벤트 (Launchpool / Earn / 입금 보너스 / 신규 상장 / 거래 대회 / 펀딩 등)
- 참여 조건 (자산, 락업 기간, 최소 금액 등)
- 보상 (APR, 보너스 금액, 토큰량 등)
- 마감일 또는 시작일

규칙:
- 사실만. 본문에 없는 정보 추측 금지
- 사견 / 평가 금지
- 짧게

Title: {title}

Body:
{body}
"""

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_SCRIPT_BLOCK_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.+?)</script>',
    re.DOTALL,
)
_TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)


# --- HTTP helpers ---
def _http_request(req: urllib.request.Request, timeout: int, label: str) -> bytes:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        try:
            snippet = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            snippet = ""
        raise RuntimeError(f"{label}: HTTP {e.code} — {snippet}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"{label}: {e.__class__.__name__}: {e}") from e


# --- Binance HTML page ---
def fetch_binance_page(code: str) -> str:
    url = BINANCE_PAGE_URL.format(code=urllib.parse.quote(code, safe="-_.~"))
    # Full Chrome-style top-level navigation header set. Binance returns a JS
    # shell to bare requests but a fully rendered page when these are present.
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    raw = _http_request(req, TIMEOUT_HTML_SEC, "binance html")
    return raw.decode("utf-8", errors="replace")


# --- Body extraction ---
def _strip_html(s: str) -> str:
    s = _SCRIPT_BLOCK_RE.sub(" ", s)
    s = _STYLE_BLOCK_RE.sub(" ", s)
    s = _HTML_TAG_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def _walk_for_article(node, depth: int = 0):
    """Depth-first search for {body|content: <html string>, title: <string>} dict
    inside arbitrary JSON. Returns (title, plain_text_body) or (None, None).
    """
    if depth > 12:  # bound recursion
        return None, None
    if isinstance(node, dict):
        for body_key in ("body", "content"):
            body = node.get(body_key)
            if isinstance(body, str) and len(body) >= 100:
                text = _strip_html(body)
                if text:
                    title = node.get("title") if isinstance(node.get("title"), str) else ""
                    return title, text
        for v in node.values():
            t, b = _walk_for_article(v, depth + 1)
            if b:
                return t, b
    elif isinstance(node, list):
        for v in node:
            t, b = _walk_for_article(v, depth + 1)
            if b:
                return t, b
    return None, None


def extract_article(html: str) -> tuple[str, str]:
    """Return (title, body_text). Prefer __NEXT_DATA__ JSON; fall back to
    raw HTML strip. Raises if body cannot be located.
    """
    # Strategy 1: __NEXT_DATA__ (Next.js SSR standard)
    m = _NEXT_DATA_RE.search(html)
    if m:
        try:
            data = json.loads(m.group(1))
            title, body = _walk_for_article(data)
            if body:
                if len(body) > MAX_BODY_CHARS:
                    body = body[:MAX_BODY_CHARS] + "..."
                if not title:
                    tm = _TITLE_RE.search(html)
                    title = tm.group(1).strip() if tm else ""
                return title, body
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: full HTML strip fallback
    tm = _TITLE_RE.search(html)
    title = tm.group(1).strip() if tm else ""
    text = _strip_html(html)
    if not text:
        raise RuntimeError("could not extract any text from HTML")
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS] + "..."
    return title, text


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
    raw = _http_request(req, TIMEOUT_GEMINI_SEC, "gemini")
    resp = json.loads(raw.decode("utf-8"))
    candidates = resp.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"gemini: no candidates — {str(resp)[:300]}")
    parts = candidates[0].get("content", {}).get("parts", []) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text.strip():
        raise RuntimeError("gemini: empty text")
    return text.strip()


# --- Pipeline ---
def summarize(code: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")
    html = fetch_binance_page(code)
    title, body = extract_article(html)
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
