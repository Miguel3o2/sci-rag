import json
import urllib.parse

from lxml import etree

from src.ingest.fetch_europepmc import build_query, download, hit_count, search_pmcids
from src.ingest.parse_pmc import parse_article


def make_get(pages, oa_ids):
    """Fake Europe PMC: pages = {cursor: (pmcids, nextCursor)}."""
    calls = []

    def get(url):
        calls.append(url)
        if "/search?" in url:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            cursor = q["cursorMark"][0]
            ids, nxt = pages[cursor]
            res = [{"pmcid": i} if i else {"id": "no-pmcid"} for i in ids]
            return json.dumps({"hitCount": 5, "nextCursorMark": nxt, "resultList": {"result": res}}).encode()
        pmcid = url.split("/")[-2]
        return b"<article/>" if pmcid in oa_ids else None

    get.calls = calls
    return get


PAGES = {"*": (["PMC1", "PMC2", None], "c2"), "c2": (["PMC2", "PMC3"], "c3"), "c3": ([], "c3")}


def test_query_and_paging_dedupes_and_skips_missing_pmcid():
    q = build_query("BMC Cancer", "breast", 2018, None)
    assert 'JOURNAL:"BMC Cancer"' in q and "OPEN_ACCESS:Y" in q and "PUB_YEAR:[2018 TO 2100]" in q
    get = make_get(PAGES, {"PMC1", "PMC2", "PMC3"})
    assert hit_count(q, get) == 5
    assert list(search_pmcids(q, get, page_size=3)) == ["PMC1", "PMC2", "PMC3"]


def test_download_resumes_counts_404_and_respects_max(tmp_path):
    get = make_get(PAGES, {"PMC1", "PMC3"})  # PMC2 -> 404
    s = download("q", tmp_path, max_articles=10, sleep=0, get=get)
    assert s == {"downloaded": 2, "already_had": 0, "not_available": 1}
    assert sorted(p.name for p in tmp_path.glob("*.xml")) == ["PMC1.xml", "PMC3.xml"]
    s = download("q", tmp_path, max_articles=10, sleep=0, get=get)  # resume: nothing re-downloaded
    assert s["downloaded"] == 0 and s["already_had"] == 2
    assert not list(tmp_path.glob("*.part"))
    s = download("q", tmp_path / "b", max_articles=1, sleep=0, get=get)
    assert s["downloaded"] == 1


def test_parse_falls_back_to_filename_id():
    xml = b"""<article><front><journal-meta><journal-title-group><journal-title>J</journal-title>
    </journal-title-group></journal-meta><article-meta><title-group><article-title>T</article-title>
    </title-group></article-meta></front><body><sec><title>Intro</title><p>word word word word</p></sec></body></article>"""
    root = etree.fromstring(xml)
    assert parse_article(root, min_body_words=1) is None
    assert parse_article(root, min_body_words=1, fallback_id="PMC77")["pmcid"] == "PMC77"
