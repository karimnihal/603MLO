import requests, json, random

# ------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"
# Quick dummy row – replace with a REAL 30-value list in production
dummy_row = [random.uniform(0, 2000) for _ in range(30)]
# ------------------------------------------------------------------

payload = {"features": dummy_row}

resp = requests.post(API_URL, json=payload, timeout=10)
print("Status code:", resp.status_code)
print("Response   :", json.dumps(resp.json(), indent=2))
