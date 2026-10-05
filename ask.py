import requests
import json
import os
import dotenv

dotenv.load_dotenv(override=True)

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}
payload = {
    "model": "deepseek/deepseek-v4.1-flash",
    "messages": [
        {
            "role": "user",
            "content": "If you built the world's tallest skyscraper, what would you name it?"
        }
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())