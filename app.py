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


def url_for(path):
    u = urlparse(LEDGER_URL)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["path"] = path
    return urlunparse(u._replace(query=urlencode(q)))


def fetch(path):
    r = requests.get(url_for(path), timeout=20)
    try:
        return r.json()
    except Exception:
        return r.text


def norm(s, cur):
    """Turn a string found in a response into an API path, or None."""
    if not isinstance(s, str) or not s or len(s) > 500:
        return None
    if s.startswith("http"):
        u = urlparse(s)
        q = dict(parse_qsl(u.query))
        if "path" in q:
            return q["path"]
        if u.netloc == urlparse(LEDGER_URL).netloc and u.path.startswith("/questionData"):
            return None
        return (u.path + ("?" + u.query if u.query else "")) if u.netloc == urlparse(LEDGER_URL).netloc else None
    if s.startswith("/") and len(s) > 1:
        return s
    if s.startswith("?"):
        return cur.split("?")[0] + s
    return None


def links(obj, cur):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in NEXT_KEYS and v not in (None, "", False):
                p = norm(v, cur)
                if p:
                    out.add(p)
                elif isinstance(v, (str, int)) and not isinstance(v, bool):
                    base = cur.split("?")[0]
                    param = "page" if "page" in lk else "cursor"
                    out.add(f"{base}?{param}={v}")
            elif isinstance(v, str):
                p = norm(v, cur)
                if p:
                    out.add(p)
            else:
                out |= links(v, cur)
    elif isinstance(obj, list):
        for v in obj:
            out |= links(v, cur)
    elif isinstance(obj, str):
        p = norm(obj, cur)
        if p:
            out.add(p)
    return out


def crawl(limit=1500):
    seen, data = set(), {}
    frontier = ["/"]
    with ThreadPoolExecutor(16) as ex:
        while frontier and len(seen) < limit:
            batch = [p for p in dict.fromkeys(frontier) if p not in seen][: limit - len(seen)]
            seen.update(batch)
            frontier = []
            for p, res in zip(batch, ex.map(lambda p: _safe(p), batch)):
                if res is None:
                    continue
                data[p] = res
                frontier += [n for n in links(res, p) if n not in seen]
    return data


def _safe(p):
    try:
        return fetch(p)
    except Exception:
        return None


def shape(v, depth=0):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {shape(x, depth+1) if depth < 2 else type(x).__name__}" for k, x in list(v.items())[:15]) + "}"
    if isinstance(v, list):
        return f"list[{len(v)}] of " + (shape(v[0], depth + 1) if v else "?")
    return type(v).__name__


DATA, SUMMARY = {}, ""


def load():
    global DATA, SUMMARY
    DATA = crawl()
    lines = []
    for p, v in list(DATA.items())[:40]:
        lines.append(f"PATH {p}\n  shape: {shape(v)}\n  sample: {json.dumps(v)[:700]}")
    SUMMARY = f"{len(DATA)} paths crawled.\n" + "\n".join(lines)


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
    err = None
    for _ in range(2):
        try:
            return {"answer": clean(run(ask_llm(q, err)))}
        except Exception as e:
            err = repr(e)[:300]
    return {"answer": None, "error": err}
