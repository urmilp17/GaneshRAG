import requests

resp = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is Ganesh Hridayam ? Explain the meaning of the name Ekdanta from it."},
    timeout=120,
)
resp.raise_for_status()
data = resp.json()
print(data)
print("ANSWER:\n", data["answer"])
print("\nMODEL:", data.get("model"))
print("TOKENS:", data.get("usage"))
print("\nSOURCES:")
for i, src in enumerate(data.get("retrieval", []), 1):
    print(f"  {i}. {src.get('source')} (page {src.get('page_number')})")