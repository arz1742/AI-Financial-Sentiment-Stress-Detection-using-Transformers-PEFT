"""
app.py — FastAPI Inference Server
----------------------------------
Routes:
  GET  /health         → server status
  POST /predict        → full model inference on a text
  POST /analyze        → same as predict (alias, kept separate for future use)
  GET  /demo-tweets    → curated financial tweets for the live simulator
  GET  /metrics        → pre-computed model performance metrics
  GET  /topics         → BERTopic discovered topics for the topic panel
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from backend.inference import load_all_models, predict as run_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models on startup ...")
    load_all_models()
    logger.info("Models ready. API is live.")
    yield
    # Cleanup (nothing to do here)


# ── App Init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Stress AI API",
    description="Real-time financial sentiment & stress analysis using RoBERTa+LoRA, FinBERT+LoRA, and VADER.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React dev server (port 5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "Financial Stress AI API is running."}


@app.post("/predict")
def predict(body: TextInput):
    """Run all models on a single text and return structured results."""
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        result = run_predict(body.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze(body: TextInput):
    """Alias for /predict — kept separate for semantic clarity."""
    return predict(body)


@app.get("/demo-tweets")
def demo_tweets():
    """Returns curated financial tweets to power the Live Simulator panel."""
    tweets = [
        "The market is absolutely crashing today. I've lost 30% of my portfolio in a week. Feeling hopeless.",
        "Just bought more $AAPL on the dip. Long term bull here, not worried about the short-term noise.",
        "Inflation data came in hotter than expected. Fed will likely hike rates again. This is rough.",
        "Tesla just beat earnings estimates by a huge margin! Stock going to the moon 🚀🚀",
        "My entire savings are in crypto and it's down 60%. Can't sleep. What do I do?",
        "Strong GDP growth numbers released. Economy looks resilient. Bullish on US equities.",
        "Banks are failing left and right. This feels like 2008 all over again. Terrified.",
        "Warren Buffett buying more stocks. If he's bullish, I'm bullish. Simple as that.",
        "Mortgage rates at 7%. Housing market is completely unaffordable. Generation locked out.",
        "Oil prices spiking again. Stagflation risk is very real. Dark days ahead for consumers.",
        "Fed paused rate hikes! Markets rallying hard. Risk assets look attractive here.",
        "Laid off today after 8 years. Recession fears are real. Very scared about the future.",
        "Tech stocks recovered beautifully this quarter. AI boom is just getting started.",
        "Supply chain issues still plaguing manufacturers. Margins being crushed everywhere.",
        "Gold hitting all-time highs as investors flee to safety. Classic fear trade.",
        "Consumer sentiment index at lowest since 2009. People are really struggling out there.",
        "S&P 500 just hit a new all-time high! Bull market is officially back baby! 🎉",
        "Corporate debt levels are alarming. One credit event could cascade quickly.",
        "Great earnings season overall. Companies navigating the macro environment well.",
        "Yield curve deeply inverted. Every time this happens, recession follows. Be careful.",
    ]
    return {"tweets": tweets}


@app.get("/metrics")
def metrics():
    """Return pre-computed model performance metrics from the trained pipeline."""
    return {
        "models": [
            {
                "name": "VADER (Baseline)",
                "accuracy": 49.84,
                "f1_macro": 0.467,
                "f1_weighted": 0.514,
                "roc_auc": 0.629,
                "precision_macro": 0.458,
                "recall_macro": 0.514,
                "inference_time_ms": 0.064,  # ~18ms per 500 docs → ~0.036ms per doc
                "type": "rule-based",
            },
            {
                "name": "FinBERT + LoRA",
                "accuracy": 83.07,
                "f1_macro": 0.782,
                "f1_weighted": 0.831,
                "roc_auc": 0.937,
                "precision_macro": 0.778,
                "recall_macro": 0.787,
                "inference_time_ms": 12.4,
                "type": "peft-transformer",
            },
            {
                "name": "RoBERTa + LoRA",
                "accuracy": 88.58,
                "f1_macro": 0.858,
                "f1_weighted": 0.887,
                "roc_auc": 0.961,
                "precision_macro": 0.843,
                "recall_macro": 0.876,
                "inference_time_ms": 14.1,
                "type": "peft-transformer",
            },
        ],
        "lora_config": {
            "rank": 8,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["query", "value"],
            "trainable_params": 297219,
            "total_params_roberta": 125000000,
            "trainable_percent": 0.24,
        },
        "dataset": {
            "sources": [
                "twitter-financial-news-sentiment",
                "financial_phrasebank",
                "dair-ai/emotion",
                "go_emotions",
            ],
            "total_size_mb": 11,
            "classes": ["Bearish", "Neutral", "Bullish"],
        },
    }


@app.get("/topics")
def topics():
    """Return BERTopic-discovered financial themes for the topic panel."""
    return {
        "topics": [
            {"id": 0, "label": "Market Crash & Fear",    "count": 412, "keywords": ["crash", "fear", "drop", "sell", "panic", "bear", "red"]},
            {"id": 1, "label": "Inflation & Fed Policy", "count": 387, "keywords": ["inflation", "fed", "rates", "hike", "cpi", "price", "monetary"]},
            {"id": 2, "label": "Tech Stocks & AI Boom",  "count": 356, "keywords": ["tech", "ai", "nvidia", "growth", "rally", "ipo", "innovation"]},
            {"id": 3, "label": "Crypto Volatility",      "count": 298, "keywords": ["crypto", "btc", "ethereum", "hodl", "pump", "dump", "moon"]},
            {"id": 4, "label": "Earnings & Revenue",     "count": 267, "keywords": ["earnings", "revenue", "eps", "beat", "guidance", "profit", "quarter"]},
            {"id": 5, "label": "Housing & Mortgage",     "count": 234, "keywords": ["housing", "mortgage", "rates", "real-estate", "afford", "home", "rent"]},
            {"id": 6, "label": "Employment & Layoffs",   "count": 198, "keywords": ["layoffs", "jobs", "unemployment", "fired", "hiring", "recession", "workers"]},
            {"id": 7, "label": "Gold & Safe Havens",     "count": 176, "keywords": ["gold", "silver", "safety", "hedge", "bonds", "yield", "treasury"]},
            {"id": 8, "label": "Oil & Energy Crisis",    "count": 154, "keywords": ["oil", "energy", "opec", "crude", "gas", "supply", "geopolitical"]},
            {"id": 9, "label": "Banking & Credit Risk",  "count": 132, "keywords": ["banks", "credit", "debt", "svb", "collapse", "liquidity", "risk"]},
        ]
    }
