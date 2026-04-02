from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=token)

models = [
    os.environ.get("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Qwen/Qwen2.5-72B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-7b-it",
    "meta-llama/Llama-3.2-1B-Instruct"
]

for model in models:
    try:
        print(f"Testing {model}...")
        response = client.chat_completion(
            model=model, 
            messages=[{"role": "user", "content": "What is RAG?"}],
            max_tokens=20
        )
        print(f"SUCCESS with {model}!")
        print(response.choices[0].message.content)
        break
    except Exception as e:
        print(f"FAILED {model}: {e}")
