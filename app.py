"""Ledger Agent: crawl undocumented API at startup, answer questions via LLM-written pandas code.

Env vars:
  LEDGER_URL       full URL from the exam page (ending in path=/)
  OPENAI_API_KEY   key for an OpenAI-compatible API (AI Pipe token works)
  OPENAI_BASE_URL  default https://api.openai.com/v1  (AI Pipe: https://aipipe.org/openai/v1)
  MODEL            default gpt-4.1-mini
Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import calendar, json, os, re, threading, time
from collections import defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

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
    prep_fallback()
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


# ---- deterministic fallback (used only if the LLM path fails or times out) ----
BUSINESS_TZ = ZoneInfo("Asia/Kolkata")
ORDERS, PRODUCT_NAMES = [], []


def parse_timestamp(value: str) -> datetime:
    """Parse a timestamp; treat naive timestamps as Kolkata-local."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BUSINESS_TZ)
    return dt



def usd(order: dict[str, Any]) -> float:
    """Convert an order's amount to USD using the supplied rate table."""
    currency = str(order["currency"]).upper()

    if currency not in RATES:
        raise ValueError(f"Unknown currency: {currency}")

    return float(order["amount"]) * RATES[currency]


def parse_period(
    question: str,
) -> tuple[datetime | None, datetime | None]:
    """Return a [start, end) interval for a month/year or year."""
    q = question.lower()
    months: dict[str, int] = {}

    for index, name in enumerate(calendar.month_name):
        if name:
            months[name.lower()] = index

    for index, name in enumerate(calendar.month_abbr):
        if name:
            months[name.lower()] = index

    month_pattern = "|".join(
        re.escape(name)
        for name in sorted(months, key=len, reverse=True)
    )

    match = re.search(rf"\b({month_pattern})\s+(20\d{{2}})\b", q)

    if match:
        month = months[match.group(1)]
        year = int(match.group(2))

        start = datetime(year, month, 1, tzinfo=BUSINESS_TZ)

        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=BUSINESS_TZ)
        else:
            end = datetime(
                year, month + 1, 1, tzinfo=BUSINESS_TZ
            )

        return start, end

    match = re.search(r"\b(20\d{2})\b", q)

    if match:
        year = int(match.group(1))

        return (
            datetime(year, 1, 1, tzinfo=BUSINESS_TZ),
            datetime(year + 1, 1, 1, tzinfo=BUSINESS_TZ),
        )

    return None, None


def find_region(question: str) -> str | None:
    """
    Recognize 'from West', 'West region', 'region West',
    and standalone region names such as 'paid Central order'.
    """
    q = question.lower()

    patterns = (
        r"\b(?:from|in|for|of|within|across)\s+"
        r"(north|south|east|west|central)\b",

        r"\b(north|south|east|west|central)\s+region\b",

        r"\bregion\s+(north|south|east|west|central)\b",

        # Keep this fallback last so explicit region phrases are preferred.
        r"\b(north|south|east|west|central)\b",
    )

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match.group(1)

    return None


def rule_answer(question: str) -> Any:
    q = question.lower().strip()

    if not q:
        raise ValueError("Please provide a question.")

    refund_query = bool(
        re.search(r"\brefund(?:s|ed|ing)?\b", q)
    )

    average_query = bool(
        re.search(r"\baverage\b|\bmean\b", q)
    )

    # Count questions must ask for orders, customers, or refunds.
    # "How many US dollars is an order worth?" is a money question,
    # not a request to count orders.
    count_query = bool(
        re.search(
            r"\bhow many\s+"
            r"(?:(?:paid|refunded|voided?|cancelled?|canceled?|"
            r"total|overall|distinct|unique)\s+)*"
            r"(?:orders?|customers?|refunds?)\b",
            q,
        )
        or re.search(
            r"\b(?:count|number)\s+of\s+"
            r"(?:(?:paid|refunded|voided?|cancelled?|canceled?|"
            r"total|overall|distinct|unique)\s+)*"
            r"(?:orders?|customers?|refunds?)\b",
            q,
        )
        or re.search(r"\bcount\b", q)
    )

    # Detect unit/quantity questions, including "most sold" and
    # "best-selling". Explicit monetary wording takes precedence.
    quantity_query = bool(
        re.search(
            r"\b(units?|items?|quantities|quantity|qty|pieces)\b",
            q,
        )
        or re.search(
            r"\b(?:sold|selling)\s+(?:the\s+)?"
            r"(?:most|highest|largest|greatest)\b|"
            r"\bmost\s+sold\b|\b(?:best|top)[- ]selling\b",
            q,
        )
    )

    # Money questions can mention value, worth, currency, cost, or revenue.
    money_query = bool(
        re.search(
            r"\b(revenue|sales|spend|spent|turnover|amount|value|"
            r"worth|price|cost|usd|dollars?|euros?|eur|"
            r"rupees?|inr|sum)\b",
            q,
        )
    )

    if money_query:
        quantity_query = False

    customer_count_query = bool(
        re.search(
            r"\bhow many customers?\b|"
            r"\b(?:number|count) of "
            r"(?:distinct |unique )?customers?\b",
            q,
        )
    )

    # Identify the requested measure.
    if count_query:
        measure = "count"
    elif refund_query:
        measure = "refund"
    elif quantity_query:
        measure = "quantity"
    elif average_query or money_query:
        measure = "revenue"
    else:
        measure = "count"

    start, end = parse_period(q)
    region = find_region(q)

    # A refund-related date normally refers to the update/event date.
    # Explicit creation/order-date wording takes precedence.
    if re.search(r"\b(updated|update date|last updated)\b", q):
        date_field = "updated_at"
    elif re.search(
        r"\b(created|creation|placed|ordered|order date)\b", q
    ):
        date_field = "created_at"
    elif refund_query:
        date_field = "updated_at"
    else:
        date_field = "created_at"

    product = next(
        (p for p in PRODUCT_NAMES if p.lower() in q),
        None,
    )

    customer_match = re.search(
        r"\b(c\d{4})\b",
        q,
        re.IGNORECASE,
    )

    filtered: list[dict[str, Any]] = []

    for order in ORDERS:
        status = str(order.get("status", "")).lower()

        # Revenue and sold quantities use paid orders only.
        # Refund values use refunded orders only.
        if measure in ("revenue", "quantity") and status != "paid":
            continue

        if measure == "refund" and status != "refunded":
            continue

        # For count questions, apply an explicit status filter.
        if measure == "count":
            if refund_query and status != "refunded":
                continue

            if (
                not refund_query
                and re.search(r"\bpaid\b", q)
                and status != "paid"
            ):
                continue

            if (
                re.search(r"\bvoid(?:ed)?\b", q)
                and status != "void"
            ):
                continue

            if (
                re.search(r"\b(?:cancelled|canceled)\b", q)
                and status not in ("cancelled", "canceled")
            ):
                continue

        # Apply date filters using Kolkata business time.
        if start is not None and end is not None:
            timestamp_value = (
                order.get(date_field)
                or order.get("created_at")
            )

            dt = parse_timestamp(timestamp_value).astimezone(
                BUSINESS_TZ
            )

            if not (start <= dt < end):
                continue

        if (
            region
            and str(order.get("region", "")).lower() != region
        ):
            continue

        if product and order.get("product") != product:
            continue

        if (
            customer_match
            and str(order.get("customer", "")).lower()
            != customer_match.group(1).lower()
        ):
            continue

        filtered.append(order)

    # Detect requests grouped or ranked by product, customer, or region.
    group_field = None

    if re.search(
        r"\b(?:top|bottom)\s+\d+\s+products?\b|\bby products?\b|\bper products?\b|\beach products?\b|"
        r"\bwhich products?\b|\btop products?\b|"
        r"\bbest[- ]selling products?\b|\bproduct[- ]wise\b",
        q,
    ):
        group_field = "product"

    elif re.search(
        r"\b(?:top|bottom)\s+\d+\s+customers?\b|\bby customers?\b|\bper customers?\b|\beach customers?\b|"
        r"\bwhich customers?\b|\btop customers?\b|"
        r"\bhighest[- ]spending customers?\b|\bcustomer[- ]wise\b",
        q,
    ):
        group_field = "customer"

    elif re.search(
        r"\b(?:top|bottom)\s+\d+\s+regions?\b|\bby regions?\b|\bper regions?\b|\beach regions?\b|"
        r"\bwhich regions?\b|\btop regions?\b|\bregion[- ]wise\b",
        q,
    ):
        group_field = "region"

    rank_cue = re.search(
        r"\b(top|highest|most|best|largest|biggest|greatest|leading|"
        r"lowest|least|fewest|smallest|worst|bottom|"
        r"(?:top|best)[- ]selling|best[- ]seller)\b",
        q,
    )

    if group_field is None and rank_cue:
        if re.search(r"\b(customers?|buyers?|clients?|who)\b", q):
            if not customer_match:
                group_field = "customer"
        elif re.search(r"\b(regions?|territor(?:y|ies)|areas?)\b", q):
            if not region:
                group_field = "region"
        elif re.search(
            r"\b(products?|appliances?|items?|sellers?|selling)\b", q
        ):
            if not product:
                group_field = "product"

    def metric_for(orders: list[dict[str, Any]]) -> float:
        if measure == "count":
            if customer_count_query:
                return float(
                    len({
                        str(o.get("customer", ""))
                        for o in orders
                    })
                )

            return float(len(orders))

        if measure == "quantity":
            total_quantity = sum(
                float(o.get("qty", 0))
                for o in orders
            )

            if average_query and orders:
                return total_quantity / len(orders)

            return total_quantity

        values = [usd(o) for o in orders]
        total = sum(values)

        if average_query and values:
            return total / len(values)

        return total

    if group_field:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for order in filtered:
            group_name = str(order.get(group_field, "Unknown"))
            buckets[group_name].append(order)

        if not buckets:
            return {}

        group_values = {
            key: metric_for(value)
            for key, value in buckets.items()
        }

        top_n = re.search(r"\b(?:top|bottom)\s+(\d+)\b", q)
        wants_min = re.search(
            r"\b(lowest|least|fewest|smallest|worst|minimum|bottom)\b",
            q,
        )
        wants_max = re.search(
            r"\b(top|highest|most|best|largest|greatest|biggest|"
            r"maximum|leading)\b",
            q,
        )

        if top_n or wants_min or wants_max:
            # "Which product ...?" asks for the name itself, so return the
            # plain name (as the API spells it), not an object with a value.
            ordered = sorted(
                group_values,
                key=lambda k: (group_values[k], k),
                reverse=not wants_min,
            )

            if top_n:
                return ordered[: int(top_n.group(1))]

            return ordered[0]

        return {
            key: (
                int(value)
                if measure in ("count", "quantity")
                and float(value).is_integer()
                else round(value, 2)
            )
            for key, value in sorted(group_values.items())
        }

    if not filtered:
        return 0

    if measure == "count":
        if customer_count_query:
            return len({
                str(order.get("customer", ""))
                for order in filtered
            })

        return len(filtered)

    if measure == "quantity":
        total_quantity = sum(
            float(order.get("qty", 0))
            for order in filtered
        )

        if average_query:
            return round(total_quantity / len(filtered), 2)

        if total_quantity.is_integer():
            return int(total_quantity)

        return round(total_quantity, 2)

    total_usd = sum(usd(order) for order in filtered)

    if average_query:
        return round(total_usd / len(filtered), 2)

    return round(total_usd, 2)




def prep_fallback():
    global ORDERS, PRODUCT_NAMES
    ORDERS = DF.drop(columns=["created_ist", "updated_ist"]).to_dict("records")
    PRODUCT_NAMES = sorted({str(o["product"]) for o in ORDERS if o.get("product")}, key=len, reverse=True)


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
- Ranking wording such as top, best, top-selling, best-selling, biggest, highest, most, leading, lowest, least
  means: group, rank, and return ONLY the single winning name string (customer ids look like 'C1067'),
  unless the question asks for the amount/value too or for 'top N' (then a list of names). "Top-selling"
  or "best seller" by revenue (or with no measure given) ranks by paid USD revenue; "units" ranks by paid qty.
- Always set `result`. It must be a plain str/int/float/list/dict, never a Series or DataFrame.
"""


def ask_llm(question, err=None, timeout=8):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"DATA SUMMARY:\n{SUMMARY[:9000]}\n\nQUESTION: {question}"}]
    if err:
        msgs.append({"role": "user", "content": f"Your previous code failed: {err}\nFix it."})
    r = requests.post(f"{BASE}/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": MODEL, "messages": msgs, "temperature": 0})
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
    code = r.json()["choices"][0]["message"]["content"]
    return re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.M).strip()


def run(code):
    import pandas as pd
    env = {"df": DF.copy(), "data": DATA, "pd": pd, "json": json, "re": re}
    exec(code, env)
    return env["result"]


def clean(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        x = x.to_dict("records")
    elif isinstance(x, pd.Series):
        if len(x) == 1:
            # a one-row ranking (groupby ... head(1)) means "the winner's name"
            x = x.index[0] if not isinstance(x.index, pd.RangeIndex) else x.iloc[0]
        else:
            x = x.to_dict()
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    try:
        import numpy as np
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating): x = float(x)
        if isinstance(x, np.bool_): return bool(x)
    except Exception:
        pass
    if isinstance(x, float):
        return int(x) if x.is_integer() and abs(x) > 1e9 else round(x, 2)
    if hasattr(x, "tolist"): return x.tolist()
    if isinstance(x, (str, int, bool)) or x is None: return x
    return str(x)


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
    """LLM-written pandas first (within a ~12s budget); rule-based fallback otherwise."""
    start, err = time.time(), None
    for _ in range(2):
        left = 12 - (time.time() - start)
        if left < 3:
            break
        try:
            return clean(run(ask_llm(q, err, timeout=min(8, left))))
        except Exception as e:
            err = repr(e)[:300]
    try:
        return clean(rule_answer(q))
    except Exception:
        return None


@app.post("/answer")
@app.post("/")
async def answer(req: Request):
    q = (await req.json()).get("question", "")
    if DF is None:
        await run_in_threadpool(load)
    # run in a worker thread so simultaneous questions are answered in parallel
    return {"answer": await run_in_threadpool(solve, q)}
