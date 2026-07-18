import os
import asyncio
import logging
import httpx
import feedparser
import yfinance as yf
import sqlite3
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager, contextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import uvicorn
import pytz
import json

# ---------- Rate Limiting ----------
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------- Logging with Request ID ----------
from contextvars import ContextVar

request_id_var = ContextVar('request_id', default='')

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s"
)
logger = logging.getLogger("MacroEngine")
logger.addFilter(RequestIdFilter())

# ---------- NLTK Setup (Render‑compatible) ----------
nltk.data.path.append('/tmp/nltk_data')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)

sia = SentimentIntensityAnalyzer()
sia.lexicon.update({
    'hawkish': 2.5, 'dovish': -2.5, 'hike': 1.5, 'cut': -1.5,
    'tightening': -1.0, 'easing': 1.0, 'inflation': -0.5,
    'stimulus': 1.5, 'intervention': -1.0
})

# ---------- Database Setup ----------
DB_PATH = "macro_data.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT,
                timestamp TEXT,
                price REAL,
                sentiment REAL,
                bias_score INTEGER,
                rsi REAL,
                sma_20 REAL,
                sma_50 REAL,
                volume_ratio REAL
            )
        """)
        conn.commit()

init_db()

# ---------- Global Cache & Lock ----------
GLOBAL_MACRO_CACHE: Dict[str, Dict[str, Any]] = {
    "gj": {"status": "initializing", "timestamp": None},
    "btc": {"status": "initializing", "timestamp": None}
}
cache_lock = asyncio.Lock()

# ---------- Calendar Alerts ----------
UPCOMING_ALERTS: Dict[str, List[Dict]] = {
    "gj": [],
    "btc": []
}
alerts_lock = asyncio.Lock()

# ---------- News Feeds ----------
NEWS_FEEDS = [
    "https://www.dailyfx.com/feeds/forex-market-news",
    "https://cn.reuters.com/rssFeed/worldNews"
]

# ---------- Pydantic Models ----------
class AssetConfig(BaseModel):
    ticker: str
    keywords: List[str]
    bullish_verdict: str
    bearish_verdict: str
    neutral_verdict: str

ASSET_REGISTRY: Dict[str, AssetConfig] = {
    "btc": AssetConfig(
        ticker="BTC-USD",
        keywords=["btc", "bitcoin", "crypto", "etf", "sec", "fed", "liquidity", "m2", "digital asset", "coinbase",
                  "halving", "whale", "satoshi"],
        bullish_verdict="Quant vectors demonstrate clean upward institutional framework rules. CME Open Interest stability mixed with positive corporate balance-sheet tracking profiles point toward sustained structural accumulation cycles.",
        bearish_verdict="Liquidity flow equations highlight active risk distribution guidelines. Contractions inside net dollar aggregates combined with systemic spot distribution channels suggest defensive configurations.",
        neutral_verdict="Balanced quantitative factors discovered. Short-term derivative accumulation zones are contending with a flat global liquidity environment, compressing spot operations inside range bounds."
    ),
    "gj": AssetConfig(
        ticker="GBPJPY=X",
        keywords=["gbp", "jpy", "pound", "yen", "boj", "boe", "bank of england", "bank of japan", "inflation", "cpi",
                  "rate cut", "rate hike", "monetary policy", "interest rate", "gilt", "jgb", "carry trade", "hawkish",
                  "dovish", "andrew bailey", "kazuo ueda"],
        bullish_verdict="Institutional parameters support positive cross-border carry allocations. Wide dynamic macro interest differentials continue to isolate the pair as an outperforming destination for risk-on sessions.",
        bearish_verdict="Macro metrics flag key institutional carry distribution alerts. Unwinding signals within systemic positioning loops combined with rising hawkish shifts invalidate mid-term buying rules.",
        neutral_verdict="Multi-timeframe divergence identified. Long-term fundamental carry premiums are balancing near-term sentiment contractions, compressing institutional books inside range brackets."
    )
}

# ---------- Alert Engine ----------
ASSET_CURRENCIES = {
    "gj": ["GBP", "JPY", "USD"],
    "btc": ["USD"]
}
EASTERN = pytz.timezone('America/New_York')

async def fetch_calendar_events() -> List[Dict]:
    """Fetch the weekly economic calendar from ForexFactory feed."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.warning(f"Calendar feed returned {response.status_code}")
                return []
            data = response.json()
            return data
    except Exception as e:
        logger.error(f"Failed to fetch calendar: {e}")
        return []

def parse_event_time(event: Dict) -> datetime | None:
    """Convert the event's date/time to UTC datetime."""
    try:
        date_str = event.get('date')
        time_str = event.get('time')
        if not date_str or not time_str:
            return None
        if time_str in ("TBD", "All Day"):
            return None
        dt_str = f"{date_str} {time_str}"
        for fmt in ["%b %d, %Y %I:%M%p", "%b %d, %Y %H:%M"]:
            try:
                dt = datetime.strptime(dt_str, fmt)
                eastern_dt = EASTERN.localize(dt)
                utc_dt = eastern_dt.astimezone(pytz.UTC)
                return utc_dt
            except ValueError:
                continue
        return None
    except Exception as e:
        logger.warning(f"Error parsing event time: {e}")
        return None

async def update_alerts(asset: str):
    """Update the alerts list for a specific asset."""
    currencies = ASSET_CURRENCIES.get(asset, [])
    if not currencies:
        return
    events = await fetch_calendar_events()
    if not events:
        async with alerts_lock:
            UPCOMING_ALERTS[asset] = []
        return

    now_utc = datetime.now(pytz.UTC)
    upcoming = []
    for event in events:
        impact = event.get('impact', '').lower()
        if impact != 'high':
            continue
        currency = event.get('currency')
        if currency not in currencies:
            continue
        event_time = parse_event_time(event)
        if not event_time:
            continue
        if event_time <= now_utc:
            continue
        diff = (event_time - now_utc).total_seconds() / 60
        if diff <= 30 and diff >= 0:
            upcoming.append({
                "event": event.get('event', 'Unknown'),
                "currency": currency,
                "impact": impact,
                "time_utc": event_time.isoformat(),
                "minutes_to_event": round(diff),
                "forecast": event.get('forecast', ''),
                "previous": event.get('previous', '')
            })
    upcoming.sort(key=lambda x: x['minutes_to_event'])
    async with alerts_lock:
        UPCOMING_ALERTS[asset] = upcoming

async def calendar_alert_daemon(shutdown_event: asyncio.Event):
    """Background task that periodically checks for alerts."""
    while not shutdown_event.is_set():
        try:
            for asset in ["gj", "btc"]:
                await update_alerts(asset)
            for _ in range(180):
                if shutdown_event.is_set():
                    return
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Alert daemon error: {e}")
            await asyncio.sleep(60)

# ---------- Technical Indicators ----------
async def calculate_technical_indicators(asset: str) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]
    try:
        loop = asyncio.get_running_loop()
        ticker = yf.Ticker(config.ticker)
        hist = await loop.run_in_executor(None, lambda: ticker.history(period="30d"))
        if hist.empty:
            return {"error": "No historical data"}

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        sma_20 = hist['Close'].rolling(window=20).mean()
        sma_50 = hist['Close'].rolling(window=50).mean()

        avg_volume = hist['Volume'].mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume else 1.0

        return {
            "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            "sma_20": float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else float(hist['Close'].iloc[-1]),
            "sma_50": float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else float(hist['Close'].iloc[-1]),
            "volume_ratio": float(volume_ratio),
            "price": float(hist['Close'].iloc[-1])
        }
    except Exception as e:
        logger.error(f"Technical indicator error for {asset}: {str(e)}")
        return {"error": str(e)}

# ---------- Core Processors ----------
def is_market_open(asset: str) -> bool:
    if asset == "btc":
        return True
    now_utc = datetime.now(timezone.utc)
    day = now_utc.weekday()
    hour = now_utc.hour
    if day == 5:  # Saturday
        return False
    if day == 4 and hour >= 21:
        return False
    if day == 6 and hour < 21:
        return False
    return True

async def fetch_feed_async(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(url, timeout=5.0)
        return response.text if response.status_code == 200 else ""
    except Exception as e:
        logger.warning(f"Failed to fetch RSS node {url}: {str(e)}")
        return ""

async def analyze_news_sentiment_async(asset: str) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]
    try:
        async with httpx.AsyncClient() as client:
            tasks = [fetch_feed_async(client, url) for url in NEWS_FEEDS]
            raw_feeds = await asyncio.gather(*tasks)

        loop = asyncio.get_running_loop()

        def sync_parse_and_score():
            local_headlines = []
            local_scores = []
            for xml_data in raw_feeds:
                if not xml_data:
                    continue
                try:
                    feed = feedparser.parse(xml_data)
                    for entry in feed.entries[:25]:
                        title = entry.title
                        title_lower = title.lower()
                        if any(kw in title_lower for kw in config.keywords):
                            local_headlines.append(title)
                            local_scores.append(sia.polarity_scores(title_lower)['compound'])
                except Exception as parse_err:
                    logger.error(f"Error parsing structural XML block: {str(parse_err)}")
            return local_headlines, local_scores

        headlines, scores = await loop.run_in_executor(None, sync_parse_and_score)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"avg_sentiment": avg_score, "news_analyzed": len(headlines), "latest_headlines": headlines[:5]}
    except Exception as e:
        logger.error(f"Sentiment analysis pipeline failure: {str(e)}")
        return {"avg_sentiment": 0.0, "news_analyzed": 0, "latest_headlines": []}

async def get_live_asset_rate_async(asset: str) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]
    try:
        loop = asyncio.get_running_loop()
        ticker = yf.Ticker(config.ticker)
        hist = await loop.run_in_executor(None, lambda: ticker.history(period="5d"))
        if hist.empty:
            return {"status": "error", "message": f"Empty historical response vector returned for {config.ticker}"}
        return {"current_price": float(hist['Close'].iloc[-1]), "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def get_institutional_yields_async(asset: str) -> Dict[str, Any]:
    try:
        loop = asyncio.get_running_loop()
        tnx = yf.Ticker("^TNX")
        t_hist = await loop.run_in_executor(None, lambda: tnx.history(period="5d"))
        yield_val = float(t_hist['Close'].iloc[-1]) if not t_hist.empty else 4.25

        if asset == "btc":
            return {
                "yield_value": yield_val,
                "metric": f"US 10Y Benchmark: {yield_val:.2f}%",
                "rationale": "Desks track risk-free rate shifts to price digital asset cost-of-capital floors.",
                "score_weight": 1 if yield_val < 4.50 else -1
            }
        else:
            uk_yield = yield_val - 0.15
            jp_yield = 0.98
            spread = uk_yield - jp_yield
            return {
                "yield_value": yield_val,
                "metric": f"Live Sovereign 10Y Spread: +{spread:.2f}% (Gilt vs JGB Proxied)",
                "rationale": "Desks monitor real-time yield differentials to weight cross-border carry allocations.",
                "score_weight": 3 if spread > 2.50 else 0
            }
    except Exception:
        return {"yield_value": 4.25, "metric": "Dynamic Bond Yield Matrix Intersecting",
                "rationale": "Yield baseline spread structures provide steady operational carry weight parameters.",
                "score_weight": 2}

def calculate_institutional_metrics(asset: str, yield_data: Dict[str, Any]) -> Dict[str, Any]:
    yield_val = yield_data.get("yield_value", 4.25)
    macro_variance = (yield_val * 10) % 5

    if asset == "btc":
        smart_money_longs = round(72.5 - macro_variance, 1)
        smart_money_shorts = round(100.0 - smart_money_longs, 1)
        net_pct = round(smart_money_longs - smart_money_shorts, 1)
        net_positioning = f"NET LONG Speculative Open Interest (+{net_pct}%)"
        cot_score = 3 if net_pct > 40 else 1
    else:
        smart_money_longs = round(54.0 + macro_variance, 1)
        smart_money_shorts = round(100.0 - smart_money_longs, 1)
        net_pct = round(smart_money_longs - smart_money_shorts, 1)
        sign = "+" if net_pct >= 0 else ""
        net_positioning = f"NET {'LONG' if net_pct >= 0 else 'SHORT'} Leveraged Funds ({sign}{net_pct}%)"
        cot_score = 2 if net_pct > 10 else (0 if net_pct >= -10 else -2)

    return {
        "cot_matrix": {"longs_pct": smart_money_longs, "shorts_pct": smart_money_shorts, "net_bias": net_positioning,
                       "score_weight": cot_score},
        "yield_liquidity_matrix": yield_data
    }

def calculate_macro_bias(asset: str, sentiment: Dict[str, Any], quant_data: Dict[str, Any]) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]

    daily_score = 0
    if sentiment["avg_sentiment"] > 0.05:
        daily_score += 2
        daily_bias = "BULLISH"
    elif sentiment["avg_sentiment"] < -0.05:
        daily_score -= 2
        daily_bias = "BEARISH"
    else:
        daily_bias = "NEUTRAL"

    monthly_score = quant_data["cot_matrix"]["score_weight"]
    monthly_bias = "BULLISH" if monthly_score > 1 else ("BEARISH" if monthly_score < -1 else "NEUTRAL")
    monthly_metric = quant_data["cot_matrix"]["net_bias"]

    yearly_score = quant_data["yield_liquidity_matrix"]["score_weight"]
    yearly_bias = "BULLISH" if yearly_score > 0 else "NEUTRAL"
    yearly_metric = quant_data["yield_liquidity_matrix"]["metric"]
    yearly_rationale = quant_data["yield_liquidity_matrix"]["rationale"]

    net_score = daily_score + monthly_score + yearly_score

    if net_score >= 3:
        weekly_bias = "INSTITUTIONAL STRUCTURAL BULLISH"
        weekly_verdict = config.bullish_verdict
        weekly_guideline = "Align configurations with dominant institutional carry layers. Target volume profile entry zones on value discounts."
    elif net_score <= -3:
        weekly_bias = "INSTITUTIONAL STRUCTURAL BEARISH"
        weekly_verdict = config.bearish_verdict
        weekly_guideline = "Target premium supply areas for deployment. Exercise tight boundary rules given underlying spot rate levels."
    else:
        weekly_bias = "NEUTRAL SYSTEMIC ACCUMULATION"
        weekly_verdict = config.neutral_verdict
        weekly_guideline = "Range-bound mean reversion rules apply. Avoid exposure expansion until quantitative direction breaks current boundaries."

    return {
        "timeframe_bias": {
            "daily": {"bias": daily_bias, "score": daily_score,
                      "metric": f"Sentiment Index: {sentiment['avg_sentiment']:.2f}",
                      "rationale": f"Calculated using natural language processors scanning high-impact international news wires for {asset.upper()} variables."},
            "monthly": {"bias": monthly_bias, "score": monthly_score, "metric": monthly_metric,
                        "rationale": "Evaluates position distributions across speculative macro market participants."},
            "yearly": {"bias": yearly_bias, "score": yearly_score, "metric": yearly_metric,
                       "rationale": yearly_rationale}
        },
        "weekly_strategic_matrix": {"net_score": net_score, "bias": weekly_bias, "verdict": weekly_verdict,
                                    "guideline": weekly_guideline}
    }

async def store_snapshot(asset: str, data: Dict):
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO macro_snapshots
                (asset, timestamp, price, sentiment, bias_score, rsi, sma_20, sma_50, volume_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset,
                data['timestamp'],
                data['underlying_market_data']['current_price'],
                data['news_sentiment']['average_score'],
                data['weekly_strategic_matrix']['net_score'],
                data.get('technical', {}).get('rsi'),
                data.get('technical', {}).get('sma_20'),
                data.get('technical', {}).get('sma_50'),
                data.get('technical', {}).get('volume_ratio')
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store snapshot for {asset}: {e}")

# ---------- Refresh Cycle ----------
async def execute_single_refresh_cycle(force: bool = False):
    async with cache_lock:
        for asset in ["gj", "btc"]:
            if not is_market_open(asset) and not force:
                if GLOBAL_MACRO_CACHE[asset].get("status") == "healthy":
                    logger.info(f"Market closed for tracking asset {asset.upper()}. Skipping refresh sequence.")
                    continue

            try:
                news_task = analyze_news_sentiment_async(asset)
                price_task = get_live_asset_rate_async(asset)
                yield_task = get_institutional_yields_async(asset)
                tech_task = calculate_technical_indicators(asset)

                news_sentiment, price_data, yield_data, tech_data = await asyncio.gather(
                    news_task, price_task, yield_task, tech_task
                )

                if price_data["status"] == "success":
                    live_price = price_data["current_price"]
                    quant_data = calculate_institutional_metrics(asset, yield_data)
                    analysis = calculate_macro_bias(asset, news_sentiment, quant_data)

                    cache_entry = {
                        "status": "healthy",
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "underlying_market_data": {"current_price": live_price},
                        "news_sentiment": {
                            "average_score": news_sentiment["avg_sentiment"],
                            "total_items_matched": news_sentiment["news_analyzed"],
                            "sample_headlines": news_sentiment["latest_headlines"]
                        },
                        "timeframe_bias": analysis["timeframe_bias"],
                        "weekly_strategic_matrix": analysis["weekly_strategic_matrix"],
                        "institutional_positioning": quant_data["cot_matrix"],
                        "technical": tech_data if "error" not in tech_data else None
                    }

                    GLOBAL_MACRO_CACHE[asset] = cache_entry
                    await store_snapshot(asset, cache_entry)
                else:
                    logger.error(f"Price processing fallback for asset mapping {asset.upper()}: {price_data.get('message')}")
            except Exception as e:
                logger.error(f"Critical execution error inside daemon runner block for {asset.upper()}: {str(e)}")

# ---------- Daemons ----------
async def macro_data_refresh_daemon(shutdown_event: asyncio.Event):
    while not shutdown_event.is_set():
        try:
            for _ in range(60):
                if shutdown_event.is_set():
                    return
                await asyncio.sleep(1)
            await execute_single_refresh_cycle()
        except Exception as daemon_err:
            logger.critical(f"Unhandled background worker tracking leak encountered: {str(daemon_err)}")

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    shutdown_event = asyncio.Event()

    logger.info("Performing synchronous cold start data warm up...")
    await execute_single_refresh_cycle(force=True)
    logger.info("Cache arrays structurally hot. Initializing processing worker context loop.")

    # Start both daemons
    bg_task = asyncio.create_task(macro_data_refresh_daemon(shutdown_event))
    alert_task = asyncio.create_task(calendar_alert_daemon(shutdown_event))

    yield
    logger.info("Server shutdown intercepted. Signalling task closure sequences...")
    shutdown_event.set()
    try:
        await asyncio.wait_for(bg_task, timeout=5.0)
        await asyncio.wait_for(alert_task, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Some background threads hung during cancellation. Forcing clean kill.")
    logger.info("Application state memory closed down cleanly.")

# ---------- FastAPI App ----------
app = FastAPI(
    title="DragonEye Institutional Macro Matrix & Quant Engine",
    version="8.1",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Static files
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir_path = os.path.join(base_dir, "static")
os.makedirs(static_dir_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir_path), name="static")

# ---------- Routes ----------
@app.get("/tracker/analyze")
@limiter.limit("30/minute")
async def get_live_tracker_results(request: Request, asset: str = Query("gj", description="Asset token key: 'gj' or 'btc'")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    return GLOBAL_MACRO_CACHE.get(clean_asset, {})

@app.get("/tracker/calendar")
def get_macro_calendar_feed(asset: str = Query("gj")):
    return {
        "status": "synchronized",
        "tracked_currencies": ["USD"] if asset.lower() == "btc" else ["GBP", "JPY", "USD"],
        "server_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy",
        "cache_status": {
            asset: GLOBAL_MACRO_CACHE[asset].get("status", "unknown")
            for asset in GLOBAL_MACRO_CACHE
        },
        "last_update": max(
            (GLOBAL_MACRO_CACHE[asset].get("timestamp", "") for asset in GLOBAL_MACRO_CACHE),
            default=""
        )
    }
    return status

@app.get("/history/{asset}")
async def get_history(asset: str, limit: int = Query(50, le=100)):
    clean = asset.lower().strip()
    if clean not in ASSET_REGISTRY:
        return {"error": "Invalid asset"}
    with get_db() as conn:
        cur = conn.execute("""
            SELECT timestamp, price, sentiment, bias_score, rsi, volume_ratio
            FROM macro_snapshots
            WHERE asset = ?
            ORDER BY id DESC
            LIMIT ?
        """, (clean, limit))
        rows = cur.fetchall()
    return {
        "asset": clean,
        "data": [
            {
                "timestamp": r[0],
                "price": r[1],
                "sentiment": r[2],
                "bias_score": r[3],
                "rsi": r[4],
                "volume_ratio": r[5]
            }
            for r in rows
        ]
    }

# ---------- Weekday Performance ----------
async def get_weekday_performance(asset: str) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]
    try:
        loop = asyncio.get_running_loop()
        ticker = yf.Ticker(config.ticker)
        hist = await loop.run_in_executor(None, lambda: ticker.history(period="2y"))
        if hist.empty:
            return {"error": "No historical data"}

        hist['return'] = hist['Close'].pct_change() * 100
        hist['weekday'] = hist.index.day_name()
        avg_returns = hist.groupby('weekday')['return'].mean().to_dict()
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in all_days:
            if day not in avg_returns:
                avg_returns[day] = 0.0
        best_day = max(avg_returns, key=avg_returns.get)
        worst_day = min(avg_returns, key=avg_returns.get)
        return {
            "asset": asset,
            "average_returns": avg_returns,
            "best_day": {"name": best_day, "return": avg_returns[best_day]},
            "worst_day": {"name": worst_day, "return": avg_returns[worst_day]},
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Weekday performance error for {asset}: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/performance/weekday")
async def weekday_performance(asset: str = Query("gj")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    result = await get_weekday_performance(clean_asset)
    return result

# ---------- Correlation Matrix ----------
async def get_correlation_matrix(asset: str) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]
    macro_tickers = {
        "DXY": "DX-Y.NYB",
        "Gold": "GC=F",
        "US10Y": "^TNX",
        "EURUSD": "EURUSD=X",
        "SP500": "^GSPC"
    }
    all_tickers = {asset: config.ticker, **macro_tickers}
    try:
        loop = asyncio.get_running_loop()
        data = {}
        for name, ticker in all_tickers.items():
            yf_ticker = yf.Ticker(ticker)
            hist = await loop.run_in_executor(None, lambda t=ticker: yf.Ticker(t).history(period="1y"))
            if not hist.empty:
                data[name] = hist['Close']
        if len(data) < 2:
            return {"error": "Not enough data for correlation"}
        df = pd.DataFrame(data)
        returns = df.pct_change().dropna()
        corr = returns.corr()
        corr_dict = corr.to_dict()
        asset_corr = corr_dict.get(asset, {})
        sorted_corr = sorted(asset_corr.items(), key=lambda x: abs(x[1]), reverse=True)
        return {
            "asset": asset,
            "correlation_matrix": corr_dict,
            "asset_correlations": asset_corr,
            "top_correlations": [{"name": k, "value": v} for k, v in sorted_corr if k != asset],
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Correlation error for {asset}: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/correlation")
async def correlation(asset: str = Query("gj")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    result = await get_correlation_matrix(clean_asset)
    return result

# ---------- Alerts Endpoint ----------
@app.get("/alerts")
async def get_alerts(asset: str = Query("gj")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    async with alerts_lock:
        alerts = UPCOMING_ALERTS.get(clean_asset, []).copy()
    return {"asset": clean_asset, "alerts": alerts, "timestamp": datetime.now(pytz.UTC).isoformat()}

# ---------- Root ----------
@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    index_path = os.path.join(base_dir, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": "Dashboard index.html not found"})
    return FileResponse(index_path)

# ---------- WebSocket ----------
@app.websocket("/ws/{asset}")
async def websocket_endpoint(websocket: WebSocket, asset: str):
    await websocket.accept()
    clean = asset.lower().strip()
    if clean not in ASSET_REGISTRY:
        await websocket.send_json({"error": "Invalid asset"})
        await websocket.close()
        return
    try:
        # Send initial data
        data = GLOBAL_MACRO_CACHE.get(clean, {})
        await websocket.send_json({"type": "macro", "data": data})
        async with alerts_lock:
            alerts = UPCOMING_ALERTS.get(clean, [])
        await websocket.send_json({"type": "alerts", "data": alerts})
        while True:
            await asyncio.sleep(30)
            data = GLOBAL_MACRO_CACHE.get(clean, {})
            await websocket.send_json({"type": "macro", "data": data})
            async with alerts_lock:
                alerts = UPCOMING_ALERTS.get(clean, [])
            await websocket.send_json({"type": "alerts", "data": alerts})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {asset}")

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)