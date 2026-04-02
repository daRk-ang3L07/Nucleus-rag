---
title: Nucleus RAG
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🚀 Nucleus RAG: High-Performance GenAI Pipeline

A production-grade, multi-model RAG (Retrieval-Augmented Generation) system built for resilience and accuracy, featuring a **TypeScript/React** frontend and a **Python/FastAPI** backend. Architected to run efficiently on **Hugging Face Spaces**.

## 🏆 Industrial-Strength Features
- **Multi-Model Fallback Router**: Intelligent failover from **Google Gemini 2.5** to **Hugging Face (Qwen-7B)** during API quota (429) events, ensuring 100% chat availability.
- **Advanced Retrieval Pipeline**: Orchestrated via **LangChain**, combining **Hybrid Search (Keyword + Semantic)** with **Multi-Query Expansion** and **TinyBERT Cross-Encoder Reranking**.
- **Quota-Friendly "Lite Auditor"**: A custom-engineered evaluation suite that performs full RAG benchmarking (Faithfulness, Relevancy) in a single-shot AI call to survive limited API quotas.
- **TypeScript Glassmorphism UI**: A premium dashboard featuring citation accordions, real-time status polling, and markdown-rendered AI responses.
- **Dockerized Architecture**: Fully containerized for one-click deployment to any OCI-compliant cloud provider.

---

## 🛠️ Tech Stack
- **AI/LLM**: Google Gemini 2.5 Flash, LangChain, Qwen-7B (Hugging Face), Phi-3 Mini.
- **Database**: ChromaDB (Vector Store), Supabase (Chat History & Auth).
- **Backend**: FastAPI, Asynchronous Python, Background Tasks, Docker.
- **Frontend**: React 18, TypeScript, Vite, Framer Motion, TailwindCSS.

---

## 🚀 One-Click HF Spaces Setup
To deploy this project to **Hugging Face Spaces**:

1. **Create a New Space**: Select the **Docker** SDK.
2. **Configure Secrets**: Add the following variables to your Space settings:
   - `GOOGLE_API_KEY`: Your Gemini API Key.
   - `HUGGINGFACEHUB_API_TOKEN`: For reranking and model fallbacks.
   - `SUPABASE_URL` & `SUPABASE_ANON_KEY`: For chat persistence.
3. **Push to Main**: The `Dockerfile` is pre-configured to handle the multi-process environment.

---

## 🧪 AI Performance Monitoring
The system includes a dedicated `/api/v1/evaluate/` endpoint that benchmarks your RAG pipeline. It uses a specialized "Lite Auditor" prompt to survive the 20-request daily limit of experimental LLM tiers.

**Key Metrics:**
- **Faithfulness**: Mathematically proves the answer is derived ONLY from your documents.
- **Answer Relevancy**: Measures how precisely the AI addressed the user's specific intent.
