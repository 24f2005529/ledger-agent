
import json
import re
import calendar
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Acme Appliances Ledger Agent")

BASE = Path(__file__).resolve().parent
BUSINESS_TZ = ZoneInfo("Asia/Kolkata")


def load_json(filename):
    with (BASE / filename).open(encoding="utf-8") as f:
        return json.load(f)


def load_orders():
    rows = load_json("orders.json")
    latest = {}

    # Keep the most recently updated version of each order.
    for row in rows:
        order_id = row["id"]
        current = latest.get(order_id)

        if (
            current is None
            or row["updated_at"] > current["updated_at"]
        ):
            latest[order_id] = row

    return list(latest.values())


ORDERS = load_orders()
RATES = load_json("rates.json")["usd_per_unit"]

PRODUCT_NAMES = sorted(
    {order["product"] for order in ORDERS},
    key=len,
    reverse=True,
)

REGIONS = ("north", "south", "east", "west", "central")


def usd(order):
    currency = order["currency"]

    if currency not in RATES:
        raise ValueError(f"Unknown currency: {currency}")

    return float(order["amount"]) * float(RATES[currency])


def parse_period(question):
    """Return a [start, end) interval in Asia/Kolkata time."""
    q = question.lower()

    months = {
        name.lower(): index
        for index, name in enumerate(calendar.month_name)
        if name
    }

    months.update({
        name.lower(): index
        for index, name in enumerate(calendar.month_abbr)
        if name
    })

    month_pattern = "|".join(
        re.escape(name)
        for name in sorted(months, key=len, reverse=True)
    )

    match = re.search(
        rf"\b({month_pattern})\s+(20\d{{2}})\b",
        q,
    )

    if match:
        month = months[match.group(1)]
        year = int(match.group(2))

        start = datetime(
            year, month, 1, tzinfo=BUSINESS_TZ
        )

        if month == 12:
            end = datetime(
                year + 1, 1, 1, tzinfo=BUSINESS_TZ
            )
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


def find_region(question):
    """Recognize phrases such as 'from West', 'West region',
    and 'region West'."""
    q = question.lower()

    patterns = [
        r"\b(?:from|in|for|of)\s+"
        r"(north|south|east|west|central)\b",

        r"\b(north|south|east|west|central)\s+region\b",

        r"\bregion\s+"
        r"(north|south|east|west|central)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match.group(1)

    return None


def answer_question(question):
    q = question.lower().strip()

    if not q:
        raise ValueError("Please provide a question.")

    refund_query = any(
        word in q for word in ("refund", "refunded", "refunds")
    )

    revenue_query = any(
        word in q
        for word in (
            "revenue",
            "sales",
            "sold",
            "spend",
            "spent",
            "turnover",
        )
    )

    count_query = any(
        phrase in q
        for phrase in (
            "how many",
            "count",
            "number of orders",
            "number of customers",
        )
    )

    average_query = any(word in q for word in ("average", "mean"))

    # A count question must return a count even if it mentions refunds.
    if refund_query and count_query:
        measure = "count"
    elif refund_query:
        measure = "refund"
    elif revenue_query or average_query:
        measure = "revenue"
    else:
        measure = "count"

    # Select the date field.
    date_field = "updated_at" if "updated" in q else "created_at"

    start, end = parse_period(q)
    region = find_region(q)

    # Identify an exact product name, if the question includes one.
    found_product = next(
        (
            product
            for product in PRODUCT_NAMES
            if product.lower() in q
        ),
        None,
    )

    # Recognize customer IDs such as C0001.
    customer_match = re.search(r"\b(c\d{4})\b", q, re.I)

    filtered = []

    for order in ORDERS:
        status = order["status"].lower()

        # Revenue includes paid orders only.
        if measure == "revenue" and status != "paid":
            continue

        # Any refund question must use refunded orders only.
        if refund_query and status != "refunded":
            continue

        # Explicit status filters for count questions.
        if (
            measure == "count"
            and not refund_query
            and re.search(r"\bpaid\b", q)
            and status != "paid"
        ):
            continue

        if (
            measure == "count"
            and not refund_query
            and re.search(r"\bvoid(?:ed)?\b", q)
            and status != "void"
        ):
            continue

        # Filter using the requested business-calendar period.
        if start is not None:
            dt = datetime.fromisoformat(
                order[date_field].replace("Z", "+00:00")
            )

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=BUSINESS_TZ)

            dt = dt.astimezone(BUSINESS_TZ)

            if not (start <= dt < end):
                continue

        # Match region names with or without the word "region".
        if region and order["region"].lower() != region:
            continue

        if found_product and order["product"] != found_product:
            continue

        if (
            customer_match
            and order["customer"].lower()
            != customer_match.group(1).lower()
        ):
            continue

        filtered.append(order)

    # Detect questions requesting grouped or ranked results.
    group_field = None

    if any(
        phrase in q
        for phrase in (
            "by product",
            "each product",
            "per product",
            "which product",
            "top product",
            "best-selling product",
            "best selling product",
        )
    ):
        group_field = "product"

    elif any(
        phrase in q
        for phrase in (
            "by customer",
            "each customer",
            "per customer",
            "which customer",
            "top customer",
            "highest-spending customer",
            "highest spending customer",
        )
    ):
        group_field = "customer"

    elif any(
        phrase in q
        for phrase in (
            "by region",
            "each region",
            "per region",
            "top region",
            "which region",
        )
    ):
        group_field = "region"

    if group_field:
        groups = defaultdict(float)

        for order in filtered:
            if measure == "count":
                value = 1
            else:
                value = usd(order)

            groups[order[group_field]] += value

        if not groups:
            return {}

        if any(
            word in q for word in ("top", "highest", "most", "best")
        ):
            key = max(groups, key=groups.get)
            value = groups[key]

            return {
                "name": key,
                "value": (
                    int(value)
                    if measure == "count"
                    else round(value, 2)
                ),
            }

        return {
            key: (
                int(value)
                if measure == "count"
                else round(value, 2)
            )
            for key, value in sorted(groups.items())
        }

    if not filtered:
        return 0

    if measure == "count":
        if (
            "customer" in q
            and "order" not in q
            and "orders" not in q
        ):
            return len({order["customer"] for order in filtered})

        return len(filtered)

    total = sum(usd(order) for order in filtered)

    if average_query:
        return round(total / len(filtered), 2)

    return round(total, 2)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "orders_loaded": len(ORDERS),
    }


@app.post("/")
def answer(request: QuestionRequest) -> dict[str, Any]:
    try:
        result = answer_question(request.question)
        return {"answer": result}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
