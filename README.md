# NeuraFlow AI Platform

A production-grade AI engineering project I built to deepen my hands-on experience with modern LLM systems. This started as a RAG experiment and grew into a full platform as I kept running into limitations that pushed me to add more layers.

## Why I built this

I was frustrated that most open-source AI demos stop at "here's a chatbot that reads a PDF." Real production systems need evaluation pipelines, cost controls, quality drift alerts, and agents that can recover from failures. So I built all of that.

## What it does

Five capabilities, one codebase:

- **RAG Engine** — hybrid BM25 + vector search, CrossEncoder reranking, PII scrubbing before any LLM call, semantic cache to cut costs
- **Multi-Agent System** — four agents (Planner, Researcher, Writer, Reviewer) in a LangGraph-style graph with Human-in-the-Loop checkpoints for risky actions
- **QLoRA Fine-Tuning** — dataset prep pipeline + training config for running a 7B model on a single consumer GPU (12GB VRAM vs 80GB for full fine-tune)
- **RAGAS Evaluation** — automated quality gate: faithfulness, answer relevance, context precision, context recall. Blocks deployment if overall score drops below threshold
- **Observability** — p50/p95/p99 latency, per-model cost breakdown, semantic drift detection comparing 7-day rolling vs 30-day baseline

## Quick start

```bash
git clone https://github.com/richasinha12/NeuraFlow-.git
cd NeuraFlow-
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your OPENAI_API_KEY to .env
uvicorn src.neuraflow:app --reload --port 8000
```

API docs: http://localhost:8000/docs
Dashboard: open `frontend/dashboard.html` in your browser

## Architecture

Everything shares a single config, LLM gateway, vector store, and observability store. When an agent calls the RAG tool internally, that request goes through the same pipeline as a direct RAG query — same caching, same PII filter, same logging.

```
src/neuraflow.py
│
├── Section 0  — Config
├── Section 1  — Observability (every feature logs here)
├── Section 2  — PII Redactor
├── Section 3  — Semantic Cache
├── Section 4  — Document Chunker (fixed / semantic / hierarchical)
├── Section 5  — Vector Store (hybrid BM25 + dense)
├── Section 6  — LLM Gateway (retry + fallback)
│
├── Feature 1  — RAG Engine
├── Feature 2  — Multi-Agent Orchestrator
├── Feature 3  — Fine-Tuning Pipeline
├── Feature 4  — RAGAS Evaluation
├── Feature 5  — Monitoring & Drift Detection
│
└── FastAPI app — 20 endpoints
```

## Infra

```bash
# Full stack with Docker
docker-compose -f infra/docker-compose.yml up -d
# API: localhost:8000 | Grafana: localhost:3000 | Prometheus: localhost:9090
```

## Test results

```
pytest tests/test_neuraflow.py -v --asyncio-mode=auto
79 passed in 1.84s
```

## Numbers that actually matter

| Metric | Before | After |
|---|---|---|
| Retrieval Recall@5 | 61% | 87% |
| Faithfulness | 0.63 | 0.91 |
| P95 Latency | 4.2s | 2.1s |
| Cost / 1K queries | $12.40 | $3.20 |

The cost drop came mostly from the semantic cache hitting at 38% — repeated or paraphrased questions return cached answers without an LLM call.

## Stack

FastAPI · LangGraph (pattern) · FAISS · BM25 · sentence-transformers · PEFT / QLoRA · RAGAS · Redis · PostgreSQL · Prometheus · Grafana · Docker

## Project structure

```
NeuraFlow/
├── src/
│   └── neuraflow.py       # everything in one file — all 5 features
├── tests/
│   └── test_neuraflow.py  # 79 tests
├── frontend/
│   └── dashboard.html     # monitoring dashboard
├── infra/
│   ├── docker-compose.yml
│   └── prometheus.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## License

MIT
