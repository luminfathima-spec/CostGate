# CostGate: Cost-Aware LLM Gateway

CostGate is an intelligent LLM routing gateway built with FastAPI, SQLite, and the Google GenAI SDK. It maximizes cost efficiency for LLM workloads while ensuring answer quality using a multi-tiered waterfall escalation strategy:

1. **Semantic Cache** (Vector cosine similarity, accounting for embedding lookup cost)
2. **Small Model Tier** (`MODEL_SMALL`: `gemini-3.5-flash-lite`)
3. **Large Model Tier** (`MODEL_LARGE`: `gemini-3.8-flash`) - escalated if small model answer quality check fails
*(Optional: Medium Tier `MODEL_MEDIUM`: `gemini-3.7-flash` via `ENABLE_MEDIUM=true`)*

---

## Gateway Routing Strategy

CostGate defaults to **two active tiers**:
- **Small Tier**: `gemini-3.5-flash-lite` (\$0.30 input / \$2.50 output per 1M tokens)
- **Large Tier**: `gemini-3.8-flash` (\$0.75 input / \$3.75 output per 1M tokens)

> [!NOTE]
> **Why Two Active Tiers by Default?**
> In the Google Gemini pricing structure (snapshot: `PRICE_DATE=2026-10-02`), `gemini-3.7-flash` (Medium) and `gemini-3.8-flash` (Large) share identical standard pricing (\$0.75 / \$3.75 per 1M tokens). Therefore, routing to Medium offers no cost advantage over Large. By setting `ENABLE_MEDIUM=false` (default), CostGate routes directly `small -> large`. Set `ENABLE_MEDIUM=true` if you wish to include Medium in the escalation waterfall.

---

## Cost Accounting & Pricing Snapshot

> [!IMPORTANT]
> **Pricing Snapshot Notice (`PRICE_DATE=2026-10-02`)**:
> The reference prices configured in `.env` are Google AI introductory rates effective through **December 31, 2026** and are scheduled to double on **January 1, 2027**.
> **Evaluation Rule**: Every evaluation results file must explicitly record the model IDs, pricing values, and `PRICE_DATE` used during the benchmark.

CostGate strictly accounts for token expenditure as follows:
- **Thinking Tokens**: Thinking/reasoning tokens returned by Gemini models are counted as billable output tokens:
  $$\text{Output Cost} = \frac{\text{output\_tokens} + \text{thinking\_tokens}}{1{,}000{,}000} \times \text{price\_output}$$
- **Embedding Lookup Costs**: Semantic cache lookups are **not** treated as free. The query vector embedding incurs an input cost:
  $$\text{Cache Lookup Cost} = \frac{\text{query\_tokens}}{1{,}000{,}000} \times \text{price\_embedding} \quad (\text{at } \$0.20 / 1\text{M tokens})$$
- **Total Request Cost**: `Input Cost + Output Cost (+ Cache Embedding Cost if applicable)`

---

## Model Retirement Handling

If an API call to a configured model returns an HTTP `404` or indicates that the model is `"no longer available"`, CostGate immediately halts execution and raises a clear `ModelRetiredError` instructing the user to update `.env`. It **never** wastes retries on retired models.

---

## Project Structure

```
costgate/
├── app/                  # Gateway core service
│   ├── main.py           # FastAPI application
│   ├── config.py         # Type-safe settings loaded from .env
│   ├── schemas.py        # Pydantic request/response schemas
│   ├── llm_client.py     # Resilient Gemini SDK client (backoff, retry, retirement)
│   ├── cost.py           # Cost calculator per 1M tokens (thinking & embedding accounting)
│   ├── cache.py          # Semantic cache with cosine similarity
│   ├── quality.py        # Response quality verification
│   ├── router.py         # Waterfall routing engine (small -> large)
│   └── logger.py         # SQLite logging for analytics
├── eval/                 # Evaluation and benchmarking
│   ├── dataset.csv       # Test query dataset
│   ├── run_eval.py       # Benchmark evaluation script (logs model IDs & PRICE_DATE)
│   └── metrics.py        # Cost & latency metrics
├── dashboard/            # Streamlit visualization
│   └── app.py            # Analytics dashboard
├── tests/                # Pytest unit & integration tests
├── .env.example          # Environment variables template
├── requirements.txt      # Dependency specification
├── Dockerfile            # Container deployment
└── README.md             # Project documentation
```

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and fill in your Gemini API key:
```bash
cp .env.example .env
```

### 3. Running Tests
```bash
pytest tests/ -v
```

### 4. Running the Gateway
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation: `http://127.0.0.1:8000/docs`

### 5. Running the Dashboard
```bash
streamlit run dashboard/app.py
```
