"""Run locally:  LEDGER_URL='<your full URL>' python explore.py
Prints the root response and a crawl summary so you can see pagination + fields."""
import json, os
from app import fetch, crawl, shape

root = fetch("/")
print("=== ROOT ===")
print(json.dumps(root, indent=2)[:4000])
data = crawl()
print("\n=== CRAWLED", len(data), "paths ===")
for p, v in list(data.items())[:60]:
    print(p, "->", shape(v))
