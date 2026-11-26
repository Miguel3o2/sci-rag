
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
RETRY_CODES = (429, 500, 502, 503, 504)
_UA = "sci-rag-portfolio-project (personal research; contact: {contact})"
_contact = "unset"


def http_get(url: str, retries: int = 6, timeout: int = 60) -> bytes | None:
    """Return body bytes, or None on HTTP 404. Retries 429/5xx/network errors with backoff."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA.format(contact=_contact)})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in RETRY_CODES:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"giving up after {retries} attempts: {url}")


def build_query(journal: str | None, query: str | None, year_from: int | None,
                year_to: int | None) -> str:
    parts = []
    if journal:
        parts.append(f'JOURNAL:"{journal}"')
    if query:
        parts.append(f"({query})")
    parts += ["OPEN_ACCESS:Y", "HAS_FT:Y"]
    if year_from or year_to:
        parts.append(f"PUB_YEAR:[{year_from or 1900} TO {year_to or 2100}]")
    return " AND ".join(parts)


def _search_url(query: str, cursor: str, page_size: int) -> str:
    return f"{BASE}/search?" + urllib.parse.urlencode({
        "query": query, "format": "json", "pageSize": page_size,
        "cursorMark": cursor, "resultType": "lite"})


def hit_count(query: str, get: Callable = http_get) -> int:
    data = json.loads(get(_search_url(query, "*", 1)))
    return int(data.get("hitCount", 0))


def search_pmcids(query: str, get: Callable = http_get, page_size: int = 100) -> Iterator[str]:
    cursor, seen = "*", set()
    while True:
        data = json.loads(get(_search_url(query, cursor, page_size)))
        results = data.get("resultList", {}).get("result", [])
        for r in results:
            pm = r.get("pmcid")
            if pm and pm not in seen:
                seen.add(pm)
                yield pm
        nxt = data.get("nextCursorMark")
        if not results or not nxt or nxt == cursor:
            return
        cursor = nxt


def download(query: str, out: Path, max_articles: int, sleep: float = 0.2,
             get: Callable = http_get) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "already_had": 0, "not_available": 0}
    for pmcid in search_pmcids(query, get):
        if stats["downloaded"] + stats["already_had"] >= max_articles:
            break
        dest = out / f"{pmcid}.xml"
        if dest.exists():
            stats["already_had"] += 1
            continue
        body = get(f"{BASE}/{pmcid}/fullTextXML")
        time.sleep(sleep)
        if body is None:
            stats["not_available"] += 1
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(dest)  # never leaves a half-written .xml if interrupted
        stats["downloaded"] += 1
        n = stats["downloaded"]
        if n % 100 == 0:
            print(f"  downloaded {n} (+{stats['already_had']} already had, "
                  f"{stats['not_available']} not available)", flush=True)
    return stats


def main() -> None:
    global _contact
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", help='journal name, e.g. "BMC Cancer"')
    ap.add_argument("--query", help="extra Europe PMC query terms, e.g. 'breast cancer AND immunotherapy'")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--max-articles", type=int, default=3000)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between full-text requests")
    ap.add_argument("--contact", default="unset", help="your email, put in the User-Agent")
    args = ap.parse_args()
    if not (args.journal or args.query):
        ap.error("give --journal and/or --query")
    _contact = args.contact

    q = build_query(args.journal, args.query, args.year_from, args.year_to)
    print("query:", q)
    total = hit_count(q)
    print(f"hitCount: {total}")
    if total == 0:
        print("No hits. Check the journal spelling (try the full journal name) or loosen the query.")
        return
    stats = download(q, Path(args.out), args.max_articles, args.sleep)
    print("done:", stats, "->", args.out)


if __name__ == "__main__":
    main()
