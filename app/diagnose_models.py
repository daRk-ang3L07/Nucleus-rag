import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
import requests

async def test_gemini():
    print("--- Testing Gemini ---")
    api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
    models_to_test = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash-exp", "gemini-2.5-flash"]
    
    for model in models_to_test:
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
            res = await llm.ainvoke("Hi")
            print(f"✅ {model}: SUCCESS!")
        except Exception as e:
            print(f"❌ {model}: FAILED ({str(e)[:50]}...)")

def test_hf():
    print("\n--- Testing Hugging Face Free Tier ---")
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}
    models = [
        "google/gemma-2b-it",
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta"
    ]
    
    for model in models:
        url = f"https://api-inference.huggingface.co/models/{model}"
        try:
            res = requests.post(url, headers=headers, json={"inputs": "Hi"}, timeout=10)
            if res.status_code == 200:
                print(f"✅ {model}: SUCCESS!")
            else:
                print(f"❌ {model}: FAILED (HTTP {res.status_code})")
        except Exception as e:
            print(f"❌ {model}: ERROR ({str(e)})")

if __name__ == "__main__":
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
    test_hf()
    asyncio.run(test_gemini())
