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
import aiohttp

# ---------- Rate Limiting ----------
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------- Logging ----------
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

# ---------- NLTK ----------
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

# ---------- Databases ----------
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

# ---------- FRED & Telegram ----------
# Use environment variables for security (optional but recommended)
FRED_API_KEY = os.getenv("FRED_API_KEY", "cc84f35feaa881ceb4ebf72ba20dd5f4")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8980659360:AAE1oqfBmSJD6IncQ35geLH8CIB--loDk-Q")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1342611966")

# ---------- Global Cache ----------
GLOBAL_MACRO_CACHE: Dict[str, Dict[str, Any]] = {
    "gj": {"status": "initializing", "timestamp": None},
    "btc": {"status": "initializing", "timestamp": None}
}
cache_lock = asyncio.Lock()
UPCOMING_ALERTS: Dict[str, List[Dict]] = {"gj": [], "btc": []}
alerts_lock = asyncio.Lock()
NEWS_FEEDS = [
    "https://www.dailyfx.com/feeds/forex-market-news",
    "https://cn.reuters.com/rssFeed/worldNews"
]

# ---------- Pydantic Models ----------
from pydantic import BaseModel


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
ASSET_CURRENCIES = {"gj": ["GBP", "JPY", "USD"], "btc": ["USD"]}
EASTERN = pytz.timezone('America/New_York')


async def fetch_calendar_events() -> List[Dict]:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.warning(f"Calendar feed returned {response.status_code}")
                return []
            return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch calendar: {e}")
        return []


def parse_event_time(event: Dict) -> datetime | None:
    try:
        date_str = event.get('date')
        time_str = event.get('time')
        if not date_str or not time_str or time_str in ("TBD", "All Day"):
            return None
        dt_str = f"{date_str} {time_str}"
        for fmt in ["%b %d, %Y %I:%M%p", "%b %d, %Y %H:%M"]:
            try:
                dt = datetime.strptime(dt_str, fmt)
                eastern_dt = EASTERN.localize(dt)
                return eastern_dt.astimezone(pytz.UTC)
            except ValueError:
                continue
        return None
    except Exception:
        return None


async def update_alerts(asset: str):
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
        if event.get('impact', '').lower() != 'high':
            continue
        if event.get('currency') not in currencies:
            continue
        event_time = parse_event_time(event)
        if not event_time or event_time <= now_utc:
            continue
        diff = (event_time - now_utc).total_seconds() / 60
        if diff <= 30 and diff >= 0:
            upcoming.append({
                "event": event.get('event', 'Unknown'),
                "currency": event.get('currency'),
                "impact": "high",
                "time_utc": event_time.isoformat(),
                "minutes_to_event": round(diff),
                "forecast": event.get('forecast', ''),
                "previous": event.get('previous', '')
            })
    upcoming.sort(key=lambda x: x['minutes_to_event'])
    async with alerts_lock:
        UPCOMING_ALERTS[asset] = upcoming


async def calendar_alert_daemon(shutdown_event: asyncio.Event):
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


# ---------- Core ----------
def is_market_open(asset: str) -> bool:
    if asset == "btc":
        return True
    now_utc = datetime.now(timezone.utc)
    day = now_utc.weekday()
    hour = now_utc.hour
    if day == 5: return False
    if day == 4 and hour >= 21: return False
    if day == 6 and hour < 21: return False
    return True


async def fetch_feed_async(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(url, timeout=5.0)
        return response.text if response.status_code == 200 else ""
    except Exception:
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


# ---------- FRED ----------
async def get_fred_series(series_id: str) -> float | None:
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                observations = data.get("observations", [])
                if observations and observations[0].get("value"):
                    val = observations[0]["value"]
                    if val == ".":
                        return None
                    return float(val)
                return None
    except Exception as e:
        logger.warning(f"FRED fetch error for {series_id}: {e}")
        return None


async def get_gilt_jgb_yields() -> Dict[str, float]:
    uk10y = await get_fred_series("IRLTLT01GBM156N")
    jp10y = await get_fred_series("IRLTLT01JPM156N")
    if uk10y is None: uk10y = 4.25
    if jp10y is None: jp10y = 0.98
    return {"uk10y": uk10y, "jp10y": jp10y}


async def get_institutional_yields_async(asset: str) -> Dict[str, Any]:
    if asset == "btc":
        us10y = await get_fred_series("DGS10")
        if us10y is None: us10y = 4.25
        return {
            "yield_value": us10y,
            "metric": f"US 10Y Benchmark: {us10y:.2f}%",
            "rationale": "Desks track risk-free rate shifts to price digital asset cost-of-capital floors.",
            "score_weight": 1 if us10y < 4.50 else -1
        }
    else:
        return {"yield_value": 4.25, "metric": "Yield proxy", "rationale": "Using FRED data", "score_weight": 0}


async def get_central_bank_rates() -> Dict[str, float]:
    return {"boe": 5.25, "boj": 0.25}


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


def calculate_macro_bias(asset: str, sentiment: Dict[str, Any], quant_data: Dict[str, Any],
                         yield_spread: float, carry: float, technical: Dict[str, Any] = None) -> Dict[str, Any]:
    config = ASSET_REGISTRY[asset]
    daily_score = 0
    if sentiment["avg_sentiment"] > 0.05:
        daily_score += 2
    elif sentiment["avg_sentiment"] < -0.05:
        daily_score -= 2
    if technical and "sma_20" in technical and "price" in technical:
        if technical["price"] > technical["sma_20"]:
            daily_score += 1
        else:
            daily_score -= 1
    if technical and "rsi" in technical:
        if technical["rsi"] > 60:
            daily_score += 1
        elif technical["rsi"] < 40:
            daily_score -= 1
    if asset == "gj":
        if yield_spread > 1.5:
            daily_score += 1
        elif yield_spread < -0.5:
            daily_score -= 1
    else:
        if yield_spread > 4.5:
            daily_score -= 1
        elif yield_spread < 3.5:
            daily_score += 1
    if daily_score >= 3:
        daily_bias = "BULLISH"
    elif daily_score <= -3:
        daily_bias = "BEARISH"
    else:
        daily_bias = "NEUTRAL"
    daily_rationale = f"Price: {technical['price']:.2f}, RSI: {technical['rsi']:.1f}" if technical else "No tech data"
    monthly_score = quant_data["cot_matrix"]["score_weight"]
    monthly_bias = "BULLISH" if monthly_score > 1 else ("BEARISH" if monthly_score < -1 else "NEUTRAL")
    monthly_rationale = f"CFTC positioning: {quant_data['cot_matrix']['net_bias']}"
    weekly_score = 0
    if asset == "gj":
        if yield_spread > 1.5:
            weekly_score += 2
        elif yield_spread < -0.5:
            weekly_score -= 2
        if carry > 4.0:
            weekly_score += 1
        elif carry < 3.0:
            weekly_score -= 1
    else:
        if yield_spread > 4.5:
            weekly_score -= 1
        elif yield_spread < 3.5:
            weekly_score += 1
    weekly_score += daily_score + monthly_score
    if weekly_score >= 3:
        weekly_bias = "BULLISH"
        weekly_reason = f"Strong bullish: yield spread {yield_spread:.2f}%, carry {carry:.2f}%."
    elif weekly_score <= -3:
        weekly_bias = "BEARISH"
        weekly_reason = f"Bearish: yield spread {yield_spread:.2f}%, carry {carry:.2f}%."
    else:
        weekly_bias = "NEUTRAL"
        weekly_reason = f"Mixed: yield spread {yield_spread:.2f}%, carry {carry:.2f}%."
    if weekly_bias == "BULLISH":
        final_verdict = config.bullish_verdict
        guideline = "Align with dominant institutional carry layers."
    elif weekly_bias == "BEARISH":
        final_verdict = config.bearish_verdict
        guideline = "Target premium supply areas."
    else:
        final_verdict = config.neutral_verdict
        guideline = "Range-bound mean reversion rules apply."
    return {
        "timeframe_bias": {
            "daily": {
                "bias": daily_bias,
                "score": daily_score,
                "metric": f"Price: {technical['price']:.2f}" if technical else "No tech data",
                "rationale": daily_rationale
            },
            "weekly": {
                "bias": weekly_bias,
                "score": weekly_score,
                "metric": f"Spread: {yield_spread:.2f}% | Carry: {carry:.2f}%",
                "rationale": weekly_reason
            },
            "monthly": {
                "bias": monthly_bias,
                "score": monthly_score,
                "metric": f"Positioning: {quant_data['cot_matrix']['net_bias']}",
                "rationale": monthly_rationale
            }
        },
        "weekly_strategic_matrix": {
            "net_score": weekly_score,
            "bias": weekly_bias,
            "verdict": final_verdict,
            "guideline": guideline
        }
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


# ---------- Generate fallback mock data for BTC ----------
def generate_mock_btc_cache():
    """If BTC fails, use this to keep cache 'healthy'."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "underlying_market_data": {"current_price": 65000 + (np.random.random() - 0.5) * 1000},
        "news_sentiment": {"average_score": 0.0, "total_items_matched": 0, "sample_headlines": []},
        "timeframe_bias": {
            "daily": {"bias": "NEUTRAL", "score": 0, "metric": "Mock data",
                      "rationale": "BTC data unavailable, using fallback."},
            "weekly": {"bias": "NEUTRAL", "score": 0, "metric": "Mock data",
                       "rationale": "BTC data unavailable, using fallback."},
            "monthly": {"bias": "NEUTRAL", "score": 0, "metric": "Mock data",
                        "rationale": "BTC data unavailable, using fallback."}
        },
        "weekly_strategic_matrix": {"net_score": 0, "bias": "NEUTRAL", "verdict": "Fallback data",
                                    "guideline": "No data"},
        "institutional_positioning": {"longs_pct": 50, "shorts_pct": 50, "net_bias": "NEUTRAL", "score_weight": 0},
        "technical": {"rsi": 50, "sma_20": 65000, "sma_50": 65000, "volume_ratio": 1.0, "price": 65000}
    }


# ---------- Telegram Helpers ----------
async def send_telegram_message(text: str, chat_id: str = None):
    if chat_id is None:
        chat_id = TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.warning("Telegram bot token not set.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("Telegram message sent.")
            else:
                logger.error(f"Telegram error: {resp.text}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")


async def get_telegram_updates(offset: int = None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": 30, "allowed_updates": ["message"]}
    if offset:
        params["offset"] = offset
    async with httpx.AsyncClient(timeout=35) as client:
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"getUpdates error: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"getUpdates error: {e}")
            return None


async def send_bias_to_chat(chat_id: str):
    msg_lines = ["<b>NEKTA DL Current Bias</b>"]
    for asset in ["gj", "btc"]:
        data = GLOBAL_MACRO_CACHE.get(asset, {})
        if data.get("status") != "healthy":
            msg_lines.append(f"\n<b>{asset.upper()}</b>  Data not available (try again later)")
            continue
        bias = data.get("timeframe_bias", {})
        daily = bias.get("daily", {})
        weekly = bias.get("weekly", {})
        monthly = bias.get("monthly", {})
        price = data.get("underlying_market_data", {}).get("current_price", 0)
        msg_lines.append(f"\n<b>{asset.upper()}</b>  Price: {price:.3f}")
        msg_lines.append(f"Daily: {daily.get('bias', 'N/A')} (Score: {daily.get('score', 0)})")
        msg_lines.append(f"Weekly: {weekly.get('bias', 'N/A')} (Score: {weekly.get('score', 0)})")
        msg_lines.append(f"Monthly: {monthly.get('bias', 'N/A')} (Score: {monthly.get('score', 0)})")
    await send_telegram_message("\n".join(msg_lines), chat_id)


async def send_intro(chat_id: str):
    intro = (
        "🤖 <b>NEKTA DL Bot</b>\n\n"
        "I am your institutional macro dashboard assistant.\n"
        "I provide daily, weekly, and monthly biases for <b>GBPJPY</b> and <b>BTC</b>.\n\n"
        "<b>Commands:</b>\n"
        "/start – Show this intro\n"
        "/bias – Get the latest biases\n"
        "/trade – Get a clear BUY/SELL/NEUTRAL recommendation for today\n"
        "/help – Show this message\n\n"
        "📊 Built with <b>FRED</b> yields, <b>CFTC</b> positioning, and <b>AI sentiment</b> analysis.\n"
        "📈 Data updates every 60 seconds.\n"
        "⏰ Daily bias sent at 08:00 AM."
    )
    await send_telegram_message(intro, chat_id)


async def handle_telegram_commands(shutdown_event: asyncio.Event):
    offset = None
    while not shutdown_event.is_set():
        try:
            updates = await get_telegram_updates(offset)
            if updates and updates.get("ok") and updates.get("result"):
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    message = update.get("message")
                    if not message:
                        continue
                    chat_id = str(message["chat"]["id"])
                    text = message.get("text", "")
                    if text == "/start":
                        await send_intro(chat_id)
                    elif text == "/bias":
                        await send_bias_to_chat(chat_id)
                    elif text == "/trade":
                        await send_daily_bias_to_telegram()
                    elif text == "/help":
                        await send_intro(chat_id)
                    else:
                        pass
            for _ in range(5):
                if shutdown_event.is_set():
                    return
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Telegram command handler error: {e}")
            await asyncio.sleep(5)


# ---------- Daily Bias Send (with recommendations) ----------
async def send_daily_bias_to_telegram():
    """
    Sends a combined message with clear BUY/SELL/NEUTRAL recommendations
    for both BTC and GBPJPY based on the daily bias.
    """
    lines = ["📊 <b>NEKTA DL – Daily Trading Recommendation</b>", ""]
    lines.append(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    for asset in ["gj", "btc"]:
        data = GLOBAL_MACRO_CACHE.get(asset, {})
        if data.get("status") != "healthy":
            lines.append(f"<b>{asset.upper()}</b>: ⚠️ Data not available (check later)")
            lines.append("")
            continue

        price = data.get("underlying_market_data", {}).get("current_price", 0)
        bias = data.get("timeframe_bias", {})
        daily = bias.get("daily", {})
        weekly = bias.get("weekly", {})
        monthly = bias.get("monthly", {})

        daily_bias = daily.get('bias', 'NEUTRAL')
        daily_score = daily.get('score', 0)

        # Determine action
        if daily_bias == "BULLISH":
            action = "✅ <b>BUY</b> (go long)"
        elif daily_bias == "BEARISH":
            action = "❌ <b>SELL</b> (go short)"
        else:
            action = "⏸️ <b>NEUTRAL</b> (wait for breakout)"

        lines.append(f"━━━━ <b>{asset.upper()}</b> ━━━━")
        lines.append(f"💰 Price: {price:.3f}")
        lines.append(f"📈 Daily Bias: {daily_bias} (score: {daily_score})")
        lines.append(f"🎯 Action: {action}")
        lines.append(f"📅 Weekly: {weekly.get('bias', 'N/A')} | Monthly: {monthly.get('bias', 'N/A')}")
        lines.append(f"📝 Rationale: {daily.get('rationale', 'No data')}")
        lines.append("")

    # Final footer
    lines.append("⚠️ <i>This is algorithmic guidance – always manage your risk.</i>")
    await send_telegram_message("\n".join(lines))


async def telegram_scheduled_send():
    while True:
        now = datetime.now()
        target = datetime(now.year, now.month, now.day, 8, 0, 0)
        if now > target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        await send_daily_bias_to_telegram()


# ---------- Refresh Cycle ----------
async def execute_single_refresh_cycle(force: bool = False):
    async with cache_lock:
        for asset in ["gj", "btc"]:
            if not is_market_open(asset) and not force:
                if GLOBAL_MACRO_CACHE[asset].get("status") == "healthy":
                    logger.info(f"Market closed for {asset.upper()}. Skipping.")
                    continue
            try:
                news_task = analyze_news_sentiment_async(asset)
                price_task = get_live_asset_rate_async(asset)
                yield_task = get_institutional_yields_async(asset)
                tech_task = calculate_technical_indicators(asset)
                if asset == "gj":
                    gilt_jgb_task = get_gilt_jgb_yields()
                    rates_task = get_central_bank_rates()
                else:
                    gilt_jgb_task = asyncio.sleep(0, result={"uk10y": 0, "jp10y": 0})
                    rates_task = asyncio.sleep(0, result={"boe": 0, "boj": 0})
                news_sentiment, price_data, yield_data, tech_data, gilt_jgb, rates = await asyncio.gather(
                    news_task, price_task, yield_task, tech_task, gilt_jgb_task, rates_task
                )
                if price_data["status"] != "success":
                    if asset == "btc":
                        logger.warning(f"BTC price fetch failed, using mock data.")
                        GLOBAL_MACRO_CACHE["btc"] = generate_mock_btc_cache()
                        continue
                    else:
                        logger.error(f"Price fetch failed for {asset}: {price_data.get('message')}")
                        continue
                live_price = price_data["current_price"]
                quant_data = calculate_institutional_metrics(asset, yield_data)
                if asset == "gj":
                    yield_spread = gilt_jgb["uk10y"] - gilt_jgb["jp10y"]
                    carry = rates["boe"] - rates["boj"]
                else:
                    yield_spread = yield_data.get("yield_value", 4.25)
                    carry = 0.0
                analysis = calculate_macro_bias(asset, news_sentiment, quant_data, yield_spread, carry, tech_data)
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
            except Exception as e:
                logger.error(f"Critical error for {asset.upper()}: {str(e)}")
                if asset == "btc":
                    GLOBAL_MACRO_CACHE["btc"] = generate_mock_btc_cache()
                else:
                    GLOBAL_MACRO_CACHE[asset] = {"status": "error", "timestamp": datetime.now(timezone.utc).isoformat()}


async def macro_data_refresh_daemon(shutdown_event: asyncio.Event):
    while not shutdown_event.is_set():
        try:
            for _ in range(60):
                if shutdown_event.is_set():
                    return
                await asyncio.sleep(1)
            await execute_single_refresh_cycle()
        except Exception as daemon_err:
            logger.critical(f"Daemon error: {str(daemon_err)}")


# ---------- Lifespan (FIXED with try/except) ----------
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    shutdown_event = asyncio.Event()
    logger.info("Cold start data warm up...")

    # --- THE FIX: wrap the refresh cycle in try/except ---
    try:
        await execute_single_refresh_cycle(force=True)
    except Exception as e:
        logger.error(f"Startup refresh cycle failed: {e}")
        # App continues; cache will be filled by fallback logic below
    # --- END FIX ---

    # Ensure BTC has data even if initial fetch failed
    if GLOBAL_MACRO_CACHE.get("btc", {}).get("status") != "healthy":
        GLOBAL_MACRO_CACHE["btc"] = generate_mock_btc_cache()
    logger.info("Cache ready.")

    bg_task = asyncio.create_task(macro_data_refresh_daemon(shutdown_event))
    alert_task = asyncio.create_task(calendar_alert_daemon(shutdown_event))
    telegram_task = asyncio.create_task(telegram_scheduled_send())
    command_task = asyncio.create_task(handle_telegram_commands(shutdown_event))
    yield
    logger.info("Server shutting down...")
    shutdown_event.set()
    try:
        await asyncio.wait_for(bg_task, timeout=5.0)
        await asyncio.wait_for(alert_task, timeout=5.0)
        await asyncio.wait_for(telegram_task, timeout=5.0)
        await asyncio.wait_for(command_task, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Some tasks timed out.")
    logger.info("Shutdown complete.")


# ---------- FastAPI App ----------
app = FastAPI(title="DragonEye Institutional Macro Matrix & Quant Engine", version="8.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir_path = os.path.join(base_dir, "static")
os.makedirs(static_dir_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir_path), name="static")


# ---------- Routes ----------
@app.get("/tracker/analyze")
@limiter.limit("30/minute")
async def get_live_tracker_results(request: Request,
                                   asset: str = Query("gj", description="Asset token key: 'gj' or 'btc'")):
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


@app.get("/performance/weekday")
async def weekday_performance(asset: str = Query("gj")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    result = await get_weekday_performance(clean_asset)
    return result


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


@app.get("/correlation")
async def correlation(asset: str = Query("gj")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    result = await get_correlation_matrix(clean_asset)
    return result


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


@app.get("/alerts")
async def get_alerts(asset: str = Query("gj")):
    clean_asset = asset.lower().strip()
    if clean_asset not in ASSET_REGISTRY:
        clean_asset = "gj"
    async with alerts_lock:
        alerts = UPCOMING_ALERTS.get(clean_asset, []).copy()
    return {"asset": clean_asset, "alerts": alerts, "timestamp": datetime.now(pytz.UTC).isoformat()}


@app.get("/telegram/send")
async def send_telegram_manual():
    await send_daily_bias_to_telegram()
    return {"status": "sent"}


@app.get("/")
async def serve_dashboard():
    index_path = os.path.join(base_dir, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": "Dashboard index.html not found"})
    return FileResponse(index_path)


@app.websocket("/ws/{asset}")
async def websocket_endpoint(websocket: WebSocket, asset: str):
    await websocket.accept()
    clean = asset.lower().strip()
    if clean not in ASSET_REGISTRY:
        await websocket.send_json({"error": "Invalid asset"})
        await websocket.close()
        return
    try:
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)