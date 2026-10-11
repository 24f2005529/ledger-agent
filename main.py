
import json
import re
import calendar
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Acme Appliances Ledger Agent")

BASE = Path(__file__).resolve().parent


def load_json(filename):
    with (BASE / filename).open(encoding="utf-8") as f:
        return json.load(f)


def load_orders():
    rows = load_json("orders.json")
    # The most recently updated version of each order is authoritative.
    latest = {}
    for row in rows:
        order_id = row["id"]
        current = latest.get(order_id)
        if current is None or row["updated_at"] > current["updated_at"]:
            latest[order_id] = row
    return list(latest.values())


ORDERS = load_orders()
RATES = load_json("rates.json")["usd_per_unit"]


def usd(order):
    currency = order["currency"]
    if currency not in RATES:
        raise ValueError(f"Unknown currency: {currency}")
    return float(order["amount"]) * float(RATES[currency])


def parse_period(question):
    """Return a [start, end) interval in Asia/Kolkata business time."""
    q = question.lower()
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    months.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

    month_pattern = "|".join(sorted(months, key=len, reverse=True))
    match = re.search(rf"\b({month_pattern})\s+(20\d{{2}})\b", q)

    if match:
        month = months[match.group(1)]
        year = int(match.group(2))
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        # The API timestamps may have different offsets. Convert to Kolkata.
        from zoneinfo import ZoneInfo
        kolkata = ZoneInfo("Asia/Kolkata")
        return start.replace(tzinfo=kolkata), end.replace(tzinfo=kolkata)

    match = re.search(r"\b(20\d{2})\b", q)
    if match:
        year = int(match.group(1))
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Kolkata")
        return datetime(year, 1, 1, tzinfo=tz), datetime(year + 1, 1, 1, tzinfo=tz)

    return None, None


def answer_question(question):
    q = question.lower().strip()

    if not q:
        raise ValueError("Please provide a question.")

    # Determine which ledger measure is requested.
    refund_query = any(w in q for w in ("refund", "refunded"))
    revenue_query = any(w in q for w in ("revenue", "sales", "sold", "spend", "spent", "turnover"))
    count_query = any(w in q for w in ("how many", "count", "number of orders", "number of customers"))
    average_query = any(w in q for w in ("average", "mean"))
    product_query = "product" in q
    customer_query = "customer" in q
    region_query = "region" in q

    if refund_query:
        measure = "refund"
    elif revenue_query or average_query:
        measure = "revenue"
    else:
        measure = "count"

    # Select the correct date field for the question.
    date_field = "created_at"
    if "updated" in q or "update date" in q:
        date_field = "updated_at"

    start, end = parse_period(q)

    # Apply explicit filters.
    filtered = []
    for order in ORDERS:
        if measure == "revenue" and order["status"] != "paid":
            continue
        if measure == "refund" and order["status"] != "refunded":
            continue

        if start is not None:
            dt = datetime.fromisoformat(order[date_field].replace("Z", "+00:00"))
            from zoneinfo import ZoneInfo
            dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
            if not (start <= dt < end):
                continue

        region_match = re.search(
            r"\b(north|south|east|west|central)\s+region\b", q
        )
        if not region_match:
            region_match = re.search(
                r"\bregion\s+(north|south|east|west|central)\b", q
            )
        if region_match and order["region"].lower() != region_match.group(1):
            continue

        # Exact product names from the API.
        product_names = sorted(
            {o["product"] for o in ORDERS}, key=len, reverse=True
        )
        found_product = next(
            (p for p in product_names if p.lower() in q), None
        )
        if found_product and order["product"] != found_product:
            continue

        customer_match = re.search(r"\b(c\d{4})\b", q, re.I)
        if customer_match and order["customer"].lower() != customer_match.group(1).lower():
            continue

        filtered.append(order)

    # Questions comparing or ranking products/customers/regions.
    group_field = None
    if any(w in q for w in ("by product", "each product", "per product", "which product", "top product", "best-selling product", "best selling product")):
        group_field = "product"
    elif any(w in q for w in ("by customer", "each customer", "per customer", "which customer", "top customer", "highest-spending customer", "highest spending customer")):
        group_field = "customer"
    elif any(w in q for w in ("by region", "each region", "per region", "which region", "top region")):
        group_field = "region"

    if group_field:
        groups = defaultdict(float)
        for order in filtered:
            value = usd(order) if measure in ("revenue", "refund") else 1
            groups[order[group_field]] += value

        if not groups:
            return {}

        if any(w in q for w in ("top", "highest", "most", "best")):
            key = max(groups, key=groups.get)
            return {"name": key, "value": round(groups[key], 2)}

        return {k: round(v, 2) for k, v in sorted(groups.items())}

    if not filtered:
        return 0

    if measure == "count":
        if "customer" in q and "order" not in q:
            return len({o["customer"] for o in filtered})
        return len(filtered)

    total = sum(usd(order) for order in filtered)

    if average_query:
        return round(total / len(filtered), 2)

    return round(total, 2)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "orders_loaded": len(ORDERS)}


@app.post("/")
def answer(request: QuestionRequest) -> dict[str, Any]:
    try:
        result = answer_question(request.question)
        return {"answer": result}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
