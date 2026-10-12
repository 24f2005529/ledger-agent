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


DATA, SUMMARY, DF = {}, "", None
RATES = {"USD": 1.0}


def build_df():
    import pandas as pd
    rows = []
    for lab, v in DATA.items():
        if lab.startswith("/orders") and isinstance(v, dict):
            rows += v.get("orders", [])
        elif lab.startswith("/export"):
            if isinstance(v, list):
                rows += v
            elif isinstance(v, str):
                for line in v.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            pass
    rates = (DATA.get("/rates") or {}).get("usd_per_unit", {})
    RATES.update({k: float(x) for k, x in rates.items()})
    df = pd.DataFrame(rows)
    for c in ("id", "customer", "region", "product", "currency", "status"):
        df[c] = df[c].astype(str).str.strip()
    df["status"] = df["status"].str.lower()
    df["currency"] = df["currency"].str.upper()
    df["_upd"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")
    df = df.sort_values("_upd", kind="stable").drop_duplicates("id", keep="last").drop(columns="_upd")
    cr = pd.to_datetime(df["created_at"], utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
    up = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
    df["created_ist"], df["updated_ist"] = cr, up
    df["month"] = cr.dt.strftime("%Y-%m")
    df["date"] = cr.dt.strftime("%Y-%m-%d")
    df["updated_month"] = up.dt.strftime("%Y-%m")
    df["usd"] = df["amount"] * df["currency"].map(RATES)
    return df.reset_index(drop=True)


def load():
    global DATA, SUMMARY, DF
    DATA = crawl()
    DF = build_df()
    d = DF
    mism = int(((d["qty"] * d["unit_price"] - d["amount"]).abs() > 0.011).sum())
    SUMMARY = (
        f"{len(DATA)} responses crawled. DataFrame `df`: {len(d)} unique orders (deduped by id, latest updated_at kept).\n"
        f"Columns/dtypes: {dict(d.dtypes.astype(str))}\n"
        f"Rates (USD per unit): {RATES}\n"
        f"status values: {d['status'].value_counts().to_dict()}\n"
        f"regions: {sorted(d['region'].unique())}\n"
        f"products: {sorted(d['product'].unique())}\n"
        f"currencies: {d['currency'].value_counts().to_dict()}\n"
        f"distinct customers: {d['customer'].nunique()}\n"
        f"created range (IST): {d['date'].min()} .. {d['date'].max()}\n"
        f"rows where qty*unit_price != amount: {mism}\n"
        f"nulls: {d.isna().sum().to_dict()}\n"
        f"sample rows:\n{d.head(5).drop(columns=['created_ist','updated_ist']).to_string()}\n"
    )


SYSTEM = """You answer questions about Acme Appliances' order ledger using a pandas DataFrame `df`
(already cleaned: one row per order id, latest version only). Write Python that computes the answer and
assigns it to `result`. pandas is `pd`. Return ONLY code, no markdown fences.
Columns: id, customer, region, product, qty, unit_price, amount (in the row's own `currency`),
currency, status (lowercase), created_at/updated_at (raw strings), created_ist/updated_ist (tz-aware Asia/Kolkata),
month ('YYYY-MM' of created_ist), date ('YYYY-MM-DD' of created_ist), updated_month, usd (amount converted to USD).
Business rules:
- Revenue = sum of `usd` over rows with status == 'paid' only. Dates for a period (month, day, quarter, year)
  use `created_ist` (columns month/date) unless the question clearly says otherwise.
- Refunds = rows with status == 'refunded'. Refund amount = sum of `usd` for them; count = number of rows.
  For "refunded in <period>" use the created period unless the question says when the refund happened
  (then use updated_month / updated_ist).
- Match region/product names case-insensitively but return product names exactly as they appear in the data.
- Money results: float rounded to 2 decimals. Counts: int. For "top/best" questions return the name string
  (or a number if they ask for the amount). Use `result` as a plain Python number/string (or list/dict if multiple).
- Quantity questions ("units sold") use sum of qty over paid orders unless stated otherwise.
"""


def ask_llm(question, err=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"DATA SUMMARY:\n{SUMMARY[:9000]}\n\nQUESTION: {question}"}]
    if err:
        msgs.append({"role": "user", "content": f"Your previous code failed: {err}\nFix it."})
    r = requests.post(f"{BASE}/chat/completions", timeout=9,
                      headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": MODEL, "messages": msgs, "temperature": 0})
    j = r.json()
    if "choices" not in j:
        raise RuntimeError("LLM API error: " + json.dumps(j)[:300])
    code = j["choices"][0]["message"]["content"]
    return re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.M).strip()


def run(code):
    import pandas as pd
    env = {"df": DF.copy(), "data": DATA, "pd": pd, "json": json, "re": re}
    exec(code, env)
    return env["result"]


def clean(x):
    import numpy as np, pandas as pd
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] == 1 else x.to_dict("records")
    if isinstance(x, pd.Series):
        x = x.index[0] if len(x) == 1 and x.index.dtype == object and not isinstance(x.iloc[0], str) else (x.iloc[0] if len(x) == 1 else x.to_dict())
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): x = float(x)
    if isinstance(x, np.str_): return str(x)
    if isinstance(x, float): return round(x, 2)
    if isinstance(x, (pd.Timestamp,)): return str(x)
    if isinstance(x, (list, tuple)): return [clean(i) for i in x]
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
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


def solve(q):
    if DF is None:
        load()
    err, code = None, None
    for _ in range(2):
        try:
            code = ask_llm(q, err)
            return {"answer": clean(run(code))}, code
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:400]
    return {"answer": None, "error": err}, code


@app.post("/answer")
@app.post("/")
async def answer(req: Request):
    q = (await req.json()).get("question", "")
    return solve(q)[0]


@app.get("/ask")
def ask(q: str):
    out, code = solve(q)
    return {**out, "code": code}
