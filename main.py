
import calendar
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Acme Appliances Ledger Agent")

BASE = Path(__file__).resolve().parent
BUSINESS_TZ = ZoneInfo("Asia/Kolkata")


def load_json(filename: str) -> Any:
    with (BASE / filename).open(encoding="utf-8") as file:
        return json.load(file)


def parse_timestamp(value: str) -> datetime:
    """Parse a timestamp; treat naive timestamps as Kolkata-local."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BUSINESS_TZ)
    return dt


def load_orders() -> list[dict[str, Any]]:
    """Keep the latest version of each order ID."""
    rows = load_json("orders.json")
    latest: dict[str, dict[str, Any]] = {}
    latest_times: dict[str, datetime] = {}

    for row in rows:
        order_id = str(row["id"])
        updated_value = row.get("updated_at") or row.get("created_at")
        updated_time = parse_timestamp(updated_value)

        if (
            order_id not in latest
            or updated_time > latest_times[order_id]
        ):
            latest[order_id] = row
            latest_times[order_id] = updated_time

    return list(latest.values())


ORDERS = load_orders()

RATES = {
    str(currency).upper(): float(rate)
    for currency, rate in load_json("rates.json")["usd_per_unit"].items()
}

PRODUCT_NAMES = sorted(
    {
        str(order["product"])
        for order in ORDERS
        if order.get("product")
    },
    key=len,
    reverse=True,
)


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


def answer_question(question: str) -> Any:
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
            r"\bmost\s+sold\b|\bbest[- ]selling\b",
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
        r"\bby products?\b|\bper products?\b|\beach products?\b|"
        r"\bwhich products?\b|\btop products?\b|"
        r"\bbest[- ]selling products?\b|\bproduct[- ]wise\b",
        q,
    ):
        group_field = "product"

    elif re.search(
        r"\bby customers?\b|\bper customers?\b|\beach customers?\b|"
        r"\bwhich customers?\b|\btop customers?\b|"
        r"\bhighest[- ]spending customers?\b|\bcustomer[- ]wise\b",
        q,
    ):
        group_field = "customer"

    elif re.search(
        r"\bby regions?\b|\bper regions?\b|\beach regions?\b|"
        r"\bwhich regions?\b|\btop regions?\b|\bregion[- ]wise\b",
        q,
    ):
        group_field = "region"

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

        if re.search(
            r"\b(top|highest|most|best|largest|greatest)\b",
            q,
        ):
            key = max(group_values, key=group_values.get)
            value = group_values[key]

            numeric_value = (
                int(value)
                if measure in ("count", "quantity")
                and float(value).is_integer()
                else round(value, 2)
            )

            return {
                "name": key,
                "value": numeric_value,
            }

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


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "orders_loaded": len(ORDERS),
    }


@app.post("/")
def answer(request: QuestionRequest) -> dict[str, Any]:
    try:
        result = answer_question(request.question)
        return {"answer": result}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
