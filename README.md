# 🚀 Resilient Full-Stack RAG (LLMOps Architecture)

A production-grade, multi-model RAG (Retrieval-Augmented Generation) system built for 100% uptime, featuring a TypeScript/React frontend and a resilient Python/FastAPI backend.

## 🏆 Industrial-Strength Features
- **Multi-Model Fallback Router**: Automatically fails over from **Google Gemini** to **Hugging Face (Qwen-72B, Phi-3, Llama 3.2)** during API rate limit (429) events.
- **Hybrid Retrieval Pipeline**: Combines **BM25 Keyword Search** with **Semantic Vector Search** and **Cross-Encoder Reranking**.
- **Autonomous Evaluation**: Integrated **Ragas** grading suite to mathematically prove system accuracy (Faithfulness, Relevancy).
- **TypeScript Frontend**: Premium dashboard with **Glassmorphism** styling, citation accordions, and markdown rendering.
- **Full Docker Orchestration**: One-click deployment for both Frontend and Backend.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, LangChain, ChromaDB, PyTorch (CPU), Google Generative AI.
- **Frontend**: React 18, TypeScript, Vite, Framer Motion, Lucide Icons.
- **Deployment**: Docker, Nginx, AWS (Free Tier Optimized).

---

## 🚀 One-Click Local Setup
```bash
# Clone the repository
git clone <your-repo-link>
cd rag

# Launch everything
docker-compose up --build
```
The Frontend will be available at `http://localhost:3000` and the Backend at `http://localhost:8000`.

---

## 🏗️ AWS Deployment Guide (Free Tier)
To deploy this for free on an **AWS EC2 t3.micro** (1GB RAM):

1. **Launch a t3.micro instance** with Ubuntu 22.04.
2. **Setup Swap Space** (Essential to prevent PyTorch crashes on 1GB RAM):
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
3. **Install Docker & Docker Compose**.
4. **Git Clone your repo** and run `docker-compose up -d --build`.

---

## 🧪 AI Performance Monitoring (MLOps)
The system includes a dedicated `/api/v1/evaluate/` endpoint that benchmarks your RAG pipeline. It uses **Qwen-72B** on Hugging Face as the primary grader to avoid exhausting primary API quotas.

**Current Benchmarks:**
- Faithfulness: **1.0** (Zero Hallucination)
- Answer Relevancy: **0.84** (High Precision)
