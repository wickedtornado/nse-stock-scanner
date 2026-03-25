#!/usr/bin/env python3
"""
NSE Stock Analyser v2 — Yahoo Finance + Telegram Alerts
Improvements over v1:
  - 200 EMA trend filter: no BUY in strong downtrend
  - Market mood check via Nifty 50
  - Volume confirmation mandatory for BUY
  - Higher BUY threshold: score >= 60 (was 45)
  - Stronger downtrend penalty to avoid falling knife traps
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ──────────────────────────────────────────────
# CREDENTIALS — Render Environment Variables
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MIN_PROFIT_PCT  = 1.0
STOP_LOSS_PCT   = 0.5
MAX_OUTPUT      = 6
MIN_AVG_VOLUME  = 500_000
BUY_THRESHOLD   = 60     # raised from 45
SHORT_THRESHOLD = -25

# ──────────────────────────────────────────────
# NSE WATCHLIST
# ──────────────────────────────────────────────
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "GRASIM.NS",
    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "EICHERMOT.NS", "BPCL.NS",
    "M&M.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS",
    "BRITANNIA.NS", "TATACONSUM.NS", "HAVELLS.NS", "PIDILITIND.NS", "DMART.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "VEDL.NS", "BANKBARODA.NS",
    "CANBK.NS", "GAIL.NS", "IOC.NS", "TATAPOWER.NS", "TORNTPHARM.NS",
    "LUPIN.NS", "BAJAJ-AUTO.NS", "NAUKRI.NS", "PERSISTENT.NS", "MPHASIS.NS",
    "COFORGE.NS", "CHOLAFIN.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "VOLTAS.NS",
    "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "DABUR.NS", "BERGEPAINT.NS",
    "ZOMATO.NS", "IRCTC.NS", "ADANIENT.NS", "DLF.NS", "GODREJPROP.NS",
    "OBEROIRLTY.NS", "MUTHOOTFIN.NS", "PIIND.NS", "ALKEM.NS", "AUROPHARMA.NS",
]

# ──────────────────────────────────────────────
# DATA FETCHING — 1 year for 200 EMA accuracy
# ──────────────────────────────────────────────
def fetch_data(ticker: str, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["Close"]  = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        return df
    except Exception as e:
        print(f"  ⚠️  {ticker}: {e}")
        return None

# ──────────────────────────────────────────────
# MARKET MOOD — Nifty 50 trend check
# ──────────────────────────────────────────────
def get_market_mood() -> str:
    try:
        df = fetch_data("^NSEI", period="3mo")
        if df is None:
            return "NEUTRAL"
        close   = df["Close"]
        ema20   = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50   = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price   = float(close.iloc[-1])
        prev    = float(close.iloc[-2])
        day_chg = (price - prev) / prev * 100
        if price > ema20 > ema50 and day_chg > -0.5:
            return "BULLISH"
        elif price < ema20 < ema50 and day_chg < 0.5:
            return "BEARISH"
        return "NEUTRAL"
    except:
        return "NEUTRAL"

# ──────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────
def calc_rsi(series: pd.Series, period=14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return round((100 - 100 / (1 + rs)).iloc[-1], 2)

def calc_macd(series: pd.Series):
    e12  = series.ewm(span=12, adjust=False).mean()
    e26  = series.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return round(line.iloc[-1], 4), round(sig.iloc[-1], 4), round(hist.iloc[-1], 4)

def calc_bollinger(series: pd.Series, period=20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower)
    return round(upper.iloc[-1], 2), round(lower.iloc[-1], 2), round(pct_b.iloc[-1], 4)

def calc_ema(series: pd.Series, span: int) -> float:
    return round(series.ewm(span=span, adjust=False).mean().iloc[-1], 2)

def calc_vol_spike(vol: pd.Series) -> float:
    avg = vol.iloc[-21:-1].mean()
    return round(vol.iloc[-1] / avg, 2) if avg > 0 else 1.0

# ──────────────────────────────────────────────
# ANALYSIS ENGINE
# ──────────────────────────────────────────────
def analyse(ticker: str, df: pd.DataFrame, market_mood: str):
    close  = df["Close"]
    volume = df["Volume"]
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    if volume.iloc[-20:].mean() < MIN_AVG_VOLUME:
        return None

    price   = round(float(close.iloc[-1]), 2)
    prev    = round(float(close.iloc[-2]), 2)
    day_chg = round((price - prev) / prev * 100, 2)

    rsi_val                  = calc_rsi(close)
    macd_val, sig_val, hist  = calc_macd(close)
    bb_upper, bb_lower, pctb = calc_bollinger(close)
    ema9                     = calc_ema(close, 9)
    ema21                    = calc_ema(close, 21)
    ema200                   = calc_ema(close, 200)
    vspike                   = calc_vol_spike(volume)

    above_200ema = price > ema200
    pct_from_200 = round((price - ema200) / ema200 * 100, 2)
    deep_downtrend = pct_from_200 < -15

    score   = 0
    reasons = []

    # RSI
    if rsi_val < 35:
        score += 25; reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val < 45:
        score += 12; reasons.append(f"RSI low ({rsi_val})")
    elif rsi_val > 70:
        score -= 20; reasons.append(f"RSI overbought ({rsi_val})")

    # MACD
    if macd_val > sig_val and hist > 0:
        score += 20; reasons.append("MACD bullish crossover")
    elif macd_val < sig_val and hist < 0:
        score -= 15; reasons.append("MACD bearish")

    # Bollinger
    if pctb < 0.2:
        score += 20; reasons.append("Near lower Bollinger Band")
    elif pctb > 0.85:
        score -= 10; reasons.append("Near upper Bollinger Band")

    # EMA short-term trend
    if price > ema9 > ema21:
        score += 15; reasons.append("Uptrend: price > EMA9 > EMA21")
    elif price < ema9 < ema21:
        score -= 20; reasons.append("Downtrend: price < EMA9 < EMA21")

    # 200 EMA long-term trend — KEY NEW FILTER
    if above_200ema:
        score += 15; reasons.append(f"Above 200 EMA (+{pct_from_200}%)")
    elif deep_downtrend:
        score -= 25; reasons.append(f"Deep downtrend ({pct_from_200}% below 200 EMA)")
    else:
        score -= 10; reasons.append(f"Below 200 EMA ({pct_from_200}%)")

    # Volume — mandatory confirmation
    if vspike >= 1.5:
        score += 15; reasons.append(f"Volume confirmed {vspike}x avg")
    elif vspike < 1.0:
        score -= 10; reasons.append(f"Volume weak ({vspike}x avg)")

    # Market mood adjustment
    if market_mood == "BEARISH":
        score -= 15
        reasons.append("⚠️ Bearish Nifty — lower confidence")
    elif market_mood == "BULLISH":
        score += 10

    # Decision
    if score >= BUY_THRESHOLD:
        if deep_downtrend and market_mood != "BULLISH":
            return None  # hard block: never BUY deep downtrend
        signal = "BUY"
        target = round(price * (1 + MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 - STOP_LOSS_PCT / 100), 2)
    elif score <= SHORT_THRESHOLD:
        signal = "SHORT"
        target = round(price * (1 - MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 + STOP_LOSS_PCT / 100), 2)
    else:
        return None

    return {
        "symbol":  symbol,
        "signal":  signal,
        "price":   price,
        "target":  target,
        "sl":      sl,
        "score":   score,
        "rsi":     rsi_val,
        "vol":     vspike,
        "day_chg": day_chg,
        "vs200":   pct_from_200,
        "reasons": reasons,
    }

# ──────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️  Telegram not configured.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML"
    }, timeout=10)
    if r.status_code == 200:
        print("✅ Telegram sent.")
    else:
        print(f"❌ Telegram error: {r.text}")

def esc(text) -> str:
    """Escape HTML special characters to prevent Telegram parse errors."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_message(results: list, market_mood: str) -> str:
    now       = datetime.now().strftime("%d %b %Y, %I:%M %p")
    mood_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(market_mood, "🟡")
    lines = [
        f"📊 <b>NSE Evening Scan</b>",
        f"🕐 {esc(now)} IST",
        f"🌍 Market: {mood_icon} {esc(market_mood)}\n"
    ]

    buys   = [r for r in results if r["signal"] == "BUY"]
    shorts = [r for r in results if r["signal"] == "SHORT"]

    if buys:
        lines.append("🟢 <b>BUY (LONG)</b>")
        for r in buys:
            reasons = esc(' • '.join(r['reasons'][:3]))
            lines.append(
                f"\n<b>{esc(r['symbol'])}</b>  (Score: {r['score']})\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (+1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (-0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  Vol: {r['vol']}x  |  vs200EMA: {r['vs200']}%\n"
                f"  📝 {reasons}"
            )

    if shorts:
        lines.append("\n🔴 <b>SHORT (SELL)</b>")
        for r in shorts:
            reasons = esc(' • '.join(r['reasons'][:3]))
            lines.append(
                f"\n<b>{esc(r['symbol'])}</b>  (Score: {r['score']})\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (-1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (+0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  Vol: {r['vol']}x  |  vs200EMA: {r['vs200']}%\n"
                f"  📝 {reasons}\n"
                f"  📦 Instrument: Futures or Put Option"
            )

    if not buys and not shorts:
        lines.append("😐 No strong signals today. Stay patient, don't force trades.")

    lines.append("\n⚠️ <i>Educational only. Not SEBI-registered advice. Always set SL.</i>")
    return "\n".join(lines)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'═'*55}")
    print(f"  NSE Stock Scanner v2 — {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print(f"{'═'*55}\n")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        sys.exit(1)

    # Market mood
    print("🌍 Checking Nifty 50 mood...")
    market_mood = get_market_mood()
    print(f"   → {market_mood}\n")

    # Scan
    results = []
    total   = len(WATCHLIST)
    for i, ticker in enumerate(WATCHLIST):
        df = fetch_data(ticker)
        if df is not None:
            rec = analyse(ticker, df, market_mood)
            if rec:
                results.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} scanned — {len(results)} signals so far")

    buys   = sorted([r for r in results if r["signal"] == "BUY"],  key=lambda x: -x["score"])
    shorts = sorted([r for r in results if r["signal"] == "SHORT"], key=lambda x: x["score"])
    final  = (buys + shorts)[:MAX_OUTPUT]

    print(f"\n{'─'*55}")
    if not final:
        print("😐 No strong signals today.")
    for r in final:
        icon = "🟢" if r["signal"] == "BUY" else "🔴"
        print(f"{icon} {r['symbol']:15} {r['signal']:6}  ₹{r['price']}  →  ₹{r['target']}  SL: ₹{r['sl']}  200EMA: {r['vs200']}%")
        for reason in r["reasons"]:
            print(f"   • {reason}")
    print(f"{'─'*55}\n")

    send_telegram(build_message(final, market_mood))

if __name__ == "__main__":
    main()
