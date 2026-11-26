
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterable, Iterator

from lxml import etree

# Elements removed before text extraction (their tail text is preserved).
DROP_XPATHS = (
    ".//xref[@ref-type='bibr']",   # citation markers like [12]
    ".//table-wrap",
    ".//fig",
    ".//supplementary-material",
    ".//disp-formula",
)


def _drop(el: etree._Element) -> None:
    parent = el.getparent()
    if parent is None:
        return
    if el.tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def _text(el: etree._Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _walk(sec: etree._Element, path: list[str], out: list[dict]) -> None:
    title = _text(sec.find("title"))
    path = path + [title] if title else path
    paras = [t for t in (_text(p) for p in sec.findall("p")) if t]
    if paras:
        out.append({"title": " > ".join(path), "paragraphs": paras})
    for sub in sec.findall("sec"):
        _walk(sub, path, out)


def parse_article(root: etree._Element, journals: list[str] | None = None,
                  min_body_words: int = 500, fallback_id: str | None = None) -> dict | None:
    if root.tag != "article":
        found = root.find(".//article")
        if found is None:
            return None
        root = found
    front = root.find("front")
    body = root.find("body")
    if front is None or body is None:
        return None

    jm = front.find("journal-meta")
    journal = _text(jm.find(".//journal-title")) if jm is not None else ""
    if journals and not any(j.lower() in journal.lower() for j in journals):
        return None

    meta = front.find("article-meta")
    if meta is None:
        return None
    pmcid = None
    for aid in meta.findall("article-id"):
        if aid.get("pub-id-type") in ("pmc", "pmcid"):
            v = _text(aid)
            pmcid = v if v.upper().startswith("PMC") else f"PMC{v}"
            break
    if not pmcid and fallback_id and fallback_id.upper().startswith("PMC"):
        pmcid = fallback_id.upper()  # e.g. taken from the file name PMC123.xml
    if not pmcid:
        return None

    for xp in DROP_XPATHS:
        for el in root.xpath(xp):
            _drop(el)

    abstract: list[str] = []
    for ab in meta.findall("abstract"):
        if ab.get("abstract-type") in ("graphical", "teaser", "toc"):
            continue
        abstract.extend(t for t in (_text(p) for p in ab.iter("p")) if t)

    sections: list[dict] = []
    loose = [t for t in (_text(p) for p in body.findall("p")) if t]
    if loose:
        sections.append({"title": "Body", "paragraphs": loose})
    for sec in body.findall("sec"):
        _walk(sec, [], sections)

    n_words = sum(len(p.split()) for s in sections for p in s["paragraphs"])
    if n_words < min_body_words:
        return None
    return {
        "pmcid": pmcid,
        "title": _text(meta.find(".//article-title")),
        "journal": journal,
        "abstract": abstract,
        "sections": sections,
        "n_words": n_words,
    }


def iter_xml(inputs: Iterable[str]) -> Iterator[tuple[str, bytes]]:
    for inp in map(Path, inputs):
        if inp.is_dir():
            files = sorted(list(inp.rglob("*.xml")) + list(inp.rglob("*.nxml")))
            for p in files:
                yield p.name, p.read_bytes()
        elif inp.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(inp, "r|gz") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith((".xml", ".nxml")):
                        f = tf.extractfile(m)
                        if f is not None:
                            yield m.name, f.read()
        else:
            yield inp.name, inp.read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/articles.jsonl")
    ap.add_argument("--journals", nargs="*", default=None,
                    help="keep articles whose journal title contains any of these substrings")
    ap.add_argument("--max-articles", type=int, default=None)
    ap.add_argument("--min-body-words", type=int, default=500)
    args = ap.parse_args()

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    kept = seen = bad = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for name, data in iter_xml(args.inputs):
            seen += 1
            try:
                root = etree.fromstring(data, parser)
            except etree.XMLSyntaxError:
                bad += 1
                continue
            art = parse_article(root, args.journals, args.min_body_words,
                                fallback_id=Path(name).stem)
            if art is None:
                continue
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_articles and kept >= args.max_articles:
                break
            if seen % 1000 == 0:
                print(f"seen={seen} kept={kept} bad_xml={bad}", flush=True)
    print(f"done: seen={seen} kept={kept} bad_xml={bad} -> {args.out}")


if __name__ == "__main__":
    main()
