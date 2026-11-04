#!/usr/bin/env python3
"""Regenerate index.html and the README index from the explanation folders.

Each explanation is a folder with an index.html. For every folder this reads the page's <h1> (or <title>) and its
<p class="sub"> summary, estimates a reading time, and takes the created / updated dates from git history. Commits
listed in .index-ignore-revs (site-wide chores such as adding a favicon) don't count as content updates.

Usage: python3 tools/build_index.py   (run from anywhere; rewrites index.html and README.md in place)
"""

import datetime as dt
import html
import json
import math
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = "https://bobrenjc93.github.io/explanations/"
REPO = "https://github.com/bobrenjc93/explanations"

# Optional per-page tags (folder -> list). Pages without an entry simply show no tags.
TAGS = {
    "pr198214": ["PyTorch", "Dynamo", "precompile"],
    "matmul-grad-dtype": ["PyTorch", "autograd", "dtypes"],
    "okc-kernels": ["CUDA", "H200", "kernels"],
}


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout


def ignored_revs():
    path = os.path.join(ROOT, ".index-ignore-revs")
    if not os.path.exists(path):
        return set()
    return {line.split()[0] for line in open(path) if line.strip() and not line.startswith("#")}


def text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def page_info(folder, skip):
    src = open(os.path.join(ROOT, folder, "index.html")).read()
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", src, re.S) or re.search(r"<title>(.*?)</title>", src, re.S)
    sub = re.search(r'<p class="sub">(.*?)</p>', src, re.S)
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", src, flags=re.S)
    words = len(text(body).split())
    log = [line.split(" ", 1) for line in git("log", "--format=%H %aI", "--", f"{folder}/index.html").splitlines()]
    content = [(sha, when) for sha, when in log if sha not in skip] or log
    created = dt.datetime.fromisoformat(content[-1][1]) if content else None
    updated = dt.datetime.fromisoformat(content[0][1]) if content else None
    return {
        "folder": folder,
        "title": text(h1.group(1)) if h1 else folder,
        "summary": text(sub.group(1)) if sub else "",
        "minutes": max(1, math.ceil(words / 230)),
        "created": created,
        "updated": updated,
        "tags": TAGS.get(folder, []),
    }


def fmt(d):
    return f"{d:%b} {d.day}, {d.year}"


def card(p):
    created, updated = p["created"], p["updated"]
    meta = [f'<time datetime="{created:%Y-%m-%d}">{fmt(created)}</time>']
    if updated and updated.date() != created.date():
        meta.append(f'<span>Updated <time datetime="{updated:%Y-%m-%d}">{fmt(updated)}</time></span>')
    meta.append(f'<span>{p["minutes"]} min read</span>')
    tags = "".join(f'<li><button type="button" class="tag" data-tag="{html.escape(t)}">{html.escape(t)}</button></li>'
                   for t in p["tags"])
    search = " ".join([p["title"], p["summary"], *p["tags"]]).lower()
    return f"""<li class="card" data-search="{html.escape(search)}" data-tags="{html.escape('|'.join(p['tags']))}">
  <article>
    <p class="meta">{' <span class="dot" aria-hidden="true">&#183;</span> '.join(meta)}</p>
    <h2><a href="{p['folder']}/">{html.escape(p['title'])}</a></h2>
    <p class="summary">{html.escape(p['summary'])}</p>
    {f'<ul class="tags" aria-label="Tags">{tags}</ul>' if tags else ''}
  </article>
</li>"""


def build():
    skip = ignored_revs()
    folders = sorted(d for d in os.listdir(ROOT)
                     if os.path.isfile(os.path.join(ROOT, d, "index.html")) and not d.startswith("."))
    pages = sorted((page_info(f, skip) for f in folders), key=lambda p: p["created"], reverse=True)
    latest = max(p["updated"] for p in pages)
    all_tags = sorted({t for p in pages for t in p["tags"]}, key=str.lower)
    cards = "\n".join(card(p) for p in pages)
    chips = "".join(f'<li><button type="button" class="tag" data-tag="{html.escape(t)}">{html.escape(t)}</button></li>'
                    for t in all_tags)
    template = open(os.path.join(ROOT, "tools", "index.template.html")).read()
    out = (template.replace("{{CARDS}}", cards).replace("{{CHIPS}}", chips).replace("{{COUNT}}", str(len(pages)))
           .replace("{{LATEST}}", fmt(latest)).replace("{{LATEST_ISO}}", f"{latest:%Y-%m-%d}").replace("{{REPO}}", REPO))
    open(os.path.join(ROOT, "index.html"), "w").write(out)

    readme_path = os.path.join(ROOT, "README.md")
    readme = open(readme_path).read()
    lines = [f"- {fmt(p['created'])}: [{p['title']}]({SITE}{p['folder']}/)" for p in pages]
    block = "## Index\n\n" + "\n".join(lines) + "\n"
    readme = re.sub(r"## Index\n.*", lambda _: block, readme, flags=re.S)
    open(readme_path, "w").write(readme)
    print(json.dumps([{k: (str(v) if isinstance(v, dt.datetime) else v) for k, v in p.items()} for p in pages],
                     indent=1))


if __name__ == "__main__":
    build()
