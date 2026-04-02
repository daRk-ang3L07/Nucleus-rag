import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

url = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 500,
    "temperature": 0.1
}

resp = requests.post(url, headers=headers, json=payload)
print(f"Status Code: {resp.status_code}")
print(f"Response Body: {resp.text}")
