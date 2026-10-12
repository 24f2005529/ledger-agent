"""Ledger Agent: crawl undocumented API at startup, answer questions via LLM-written pandas code.

Env vars:
  LEDGER_URL       full URL from the exam page (ending in path=/)
  OPENAI_API_KEY   key for an OpenAI-compatible API (AI Pipe token works)
  OPENAI_BASE_URL  default https://api.openai.com/v1  (AI Pipe: https://aipipe.org/openai/v1)
  MODEL            default gpt-4.1-mini
Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json, os, re, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

LEDGER_URL = os.environ["LEDGER_URL"]
BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4.1-mini")
NEXT_KEYS = {"next", "next_url", "nexturl", "next_page", "nextpage", "next_cursor",
             "nextcursor", "cursor", "after", "next_token", "nexttoken"}


def set_param(url, **kw):
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q.update({k: str(v) for k, v in kw.items()})
    return urlunparse(u._replace(query=urlencode(q)))


def url_for(path):
    return set_param(LEDGER_URL, path=path)


HOST = urlparse(LEDGER_URL).netloc
AUTH = {"email", "quizSign", "questionId"}


def label(url):
    q = dict(parse_qsl(urlparse(url).query))
    extra = "&".join(f"{k}={v}" for k, v in q.items() if k not in AUTH and k != "path")
    return q.get("path", "/") + ("?" + extra if extra else "")


def fetch(x):
    url = x if x.startswith("http") else url_for(x)
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            try:
                return r.json()
            except Exception:
                return r.text
        except Exception:
            pass
    return None


def to_url(s, cur):
    if not isinstance(s, str) or not s or len(s) > 3000:
        return None
    if s.startswith("http"):
        return s if urlparse(s).netloc == HOST else None
    if s.startswith("/") and len(s) > 1:
        p, _, qs = s.partition("?")
        url = set_param(LEDGER_URL, path=p)
        for k, v in parse_qsl(qs):
            url = set_param(url, **{k: v})
        return url
    if s.startswith("?"):
        url = cur
        for k, v in parse_qsl(s[1:]):
            url = set_param(url, **{k: v})
        return url
    return None


def links(obj, cur):
    out = set()
    if isinstance(obj, dict):
        low = {str(k).lower(): v for k, v in obj.items()}
        for n in ("total_pages", "totalpages", "pages", "last_page", "page_count"):
            if isinstance(low.get(n), int) and 1 < low[n] <= 500:
                out |= {set_param(cur, page=p) for p in range(1, low[n] + 1)}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in NEXT_KEYS and v not in (None, "", False):
                u = to_url(v, cur)
                if u:
                    out.add(u)
                elif isinstance(v, (str, int)) and not isinstance(v, bool):
                    param = "page" if "page" in lk else ("after" if lk == "after" else "cursor")
                    out.add(set_param(cur, **{param: v}))
                else:
                    out |= links(v, cur)
            elif isinstance(v, str):
                u = to_url(v, cur)
                if u:
                    out.add(u)
            else:
                out |= links(v, cur)
    elif isinstance(obj, list):
        for v in obj:
            out |= links(v, cur)
    elif isinstance(obj, str):
        u = to_url(obj, cur)
        if u:
            out.add(u)
    return out


def has_records(res):
    if isinstance(res, list):
        return len(res) > 0
    if isinstance(res, dict):
        return any(isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) for v in res.values())
    return False


def crawl(limit=1500):
    seen, data = set(), {}
    frontier = [LEDGER_URL]
    with ThreadPoolExecutor(16) as ex:
        while frontier and len(seen) < limit:
            batch = [u for u in dict.fromkeys(frontier) if u not in seen][: limit - len(seen)]
            seen.update(batch)
            frontier = []
            for u, res in zip(batch, ex.map(fetch, batch)):
                if res is None:
                    continue
                data[label(u)] = res
                found = links(res, u)
                q = dict(parse_qsl(urlparse(u).query))
                if "page" in q and q["page"].isdigit() and has_records(res):
                    cp = q["page"]
                    if not any(dict(parse_qsl(urlparse(f).query)).get("page", cp) != cp for f in found):
                        found.add(set_param(u, page=int(cp) + 1))
                frontier += [f for f in found if f not in seen]
    return data


def shape(v, depth=0):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {shape(x, depth+1) if depth < 2 else type(x).__name__}" for k, x in list(v.items())[:20]) + "}"
    if isinstance(v, list):
        return f"list[{len(v)}] of " + (shape(v[0], depth + 1) if v else "?")
    return type(v).__name__


DATA, SUMMARY = {}, ""


def load():
    global DATA, SUMMARY
    DATA = crawl()
    root = DATA.get("/")
    lines = [f"{len(DATA)} responses crawled."]
    if isinstance(root, dict):
        r2 = {k: (v if k != "links" else {a: label(b) if isinstance(b, str) else b for a, b in v.items()}) for k, v in root.items()}
        lines.append("ROOT (full):\n" + json.dumps(r2)[:3000])
    groups = {}
    for lab, v in DATA.items():
        groups.setdefault(lab.split("?")[0], []).append((lab, v))
    for base, items in groups.items():
        if base == "/":
            continue
        n = sum(len(x) if isinstance(x, list) else 0 for _, x in items)
        lines.append(f"PATH {base}: {len(items)} responses (e.g. {items[0][0]})\n  shape: {shape(items[0][1])}\n"
                     f"  sample: {json.dumps(items[0][1])[:1500]}\n  last: {json.dumps(items[-1][1])[:500]}")
    SUMMARY = "\n".join(lines)


SYSTEM = """You answer questions about an order ledger. Data is in the Python dict `data`
(key = API path, value = parsed JSON response). pandas is imported as `pd`, `json` and `re` too.
Write Python that computes the answer and assigns it to a variable `result`.
Rules:
- Flatten the paginated record lists across ALL paths; DROP duplicate records (same id).
- Inspect field names/values from the summary. Respect status fields (e.g. cancelled/refunded),
  currencies (convert to USD if amounts are in other currencies using any rates in the data),
  date fields (filter by calendar month/year in the right field), and region names exactly.
- Money: round to 2 decimals. Counts: int. Names: spelled exactly as in the data.
- Refunds may live in a separate list or as fields on orders; check the summary.
Return ONLY code, no markdown fences."""


def ask_llm(question, err=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"DATA SUMMARY:\n{SUMMARY[:14000]}\n\nQUESTION: {question}"}]
    if err:
        msgs.append({"role": "user", "content": f"Your previous code failed: {err}\nFix it."})
    r = requests.post(f"{BASE}/chat/completions", timeout=12,
                      headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": MODEL, "messages": msgs, "temperature": 0})
    code = r.json()["choices"][0]["message"]["content"]
    return re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.M).strip()


def run(code):
    import pandas as pd
    env = {"data": DATA, "pd": pd, "json": json, "re": re}
    exec(code, env)
    return env["result"]


def clean(x):
    try:
        import numpy as np
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): x = float(x)
    except Exception:
        pass
    if isinstance(x, float):
        return int(x) if x.is_integer() and abs(x) > 1e9 else round(x, 2)
    if hasattr(x, "tolist"): return x.tolist()
    return x


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    load()


@app.get("/")
def health():
    return {"ok": True, "paths": len(DATA)}


@app.get("/debug")
def debug():
    return {"summary": SUMMARY}


@app.post("/answer")
@app.post("/")
async def answer(req: Request):
    q = (await req.json()).get("question", "")
    if not DATA:
        load()
    err = None
    for _ in range(2):
        try:
            return {"answer": clean(run(ask_llm(q, err)))}
        except Exception as e:
            err = repr(e)[:300]
    return {"answer": None, "error": err}
