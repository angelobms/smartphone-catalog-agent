"""Check recent main_agent_loop data via the Langfuse v2 metrics API."""
import datetime
import json
import sys

import dotenv

dotenv.load_dotenv(sys.argv[1] if len(sys.argv) > 1 else ".env")
from langfuse import get_client

end = datetime.datetime.now(datetime.timezone.utc)
start = end - datetime.timedelta(minutes=30)

query = {
    "view": "observations",
    "metrics": [{"measure": "count", "aggregation": "count"}],
    "dimensions": [
        {"field": "traceName"},
        {"field": "userId"},
        {"field": "sessionId"},
    ],
    "filters": [],
    "orderBy": [{"field": "count_count", "direction": "desc"}],
    "config": {"row_limit": 50},
    "fromTimestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "toTimestamp": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
}

res = get_client().api.metrics.metrics(query=json.dumps(query))
for row in res.data:
    print(row)
if not res.data:
    print("No data in the last 30 minutes (traces may still be indexing).")
