#!/usr/bin/env python3
"""
NSE Stock Analyser v4 — Research-backed Multi-Layer Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What's new vs v3:
  1. ADX filter — only trade stocks in strong trends (ADX > 25)
     Below 25 = ranging market = skip entirely
  2. Corrected RSI logic — BUY when RSI 35-55 in uptrend (pullback)
     Not RSI < 30 in downtrend (falling knife)
  3. FII/DII sentiment — fetched live from NSE
     Heavy FII selling = suppress BUY signals
  4. SuperTrend + EMA must BOTH agree for signal
  5. India VIX awareness — high VIX = tighter filters
  6. Cleaner signal output — quality over quantity
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ──────────────────────────────────────────────
# CREDENTIALS
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MIN_PROFIT_PCT  = 1.0
STOP_LOSS_PCT   = 0.5
MAX_OUTPUT      = 5
MIN_AVG_VOLUME  = 500_000
ADX_THRESHOLD   = 25      # below this = ranging market = skip
RSI_BUY_MIN     = 35      # RSI pullback zone for BUY in uptrend
RSI_BUY_MAX     = 55      # don't buy overbought stocks
RSI_SHORT_MIN   = 50      # don't short oversold stocks
RSI_SHORT_MAX   = 72      # RSI overbought zone for SHORT in downtrend

# ──────────────────────────────────────────────
# WATCHLIST
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
# DATA FETCHING
# ──────────────────────────────────────────────
def fetch_data(ticker: str, period: str = "1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for col in df.columns:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return None

# ──────────────────────────────────────────────
# FII/DII SENTIMENT — fetch from NSE
# ──────────────────────────────────────────────
def get_fii_sentiment() -> dict:
    """
    Fetch latest FII/DII data from NSE.
    Returns dict with net_fii (crore), net_dii (crore), sentiment label.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        r = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=headers, timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            # Get most recent entry
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                fii_net = float(latest.get("fiinet", 0))
                dii_net = float(latest.get("diinet", 0))

                if fii_net > 2000:
                    sentiment = "FII_BULLISH"
                elif fii_net < -2000 and dii_net < 0:
                    sentiment = "FII_BEARISH"
                elif fii_net < -2000:
                    sentiment = "FII_SELLING_DII_SUPPORTING"
                else:
                    sentiment = "NEUTRAL"

                return {
                    "fii_net": round(fii_net, 0),
                    "dii_net": round(dii_net, 0),
                    "sentiment": sentiment
                }
    except Exception as e:
        print(f"  ⚠️  FII data fetch failed: {e}")

    return {"fii_net": 0, "dii_net": 0, "sentiment": "NEUTRAL"}

# ──────────────────────────────────────────────
# MARKET MOOD — Nifty 50 trend (improved)
# Factors: EMA alignment + ADX strength + breadth
# ──────────────────────────────────────────────
def get_market_mood() -> dict:
    """
    Returns mood dict with:
      - mood: BULLISH / BEARISH / NEUTRAL
      - adx: Nifty trend strength
      - breadth: % of watchlist stocks above 200 EMA
      - detail: human readable summary
    """
    df = fetch_data("^NSEI")
    if df is None:
        return {"mood": "NEUTRAL", "adx": 0, "breadth": 0, "detail": "Data unavailable"}

    close = df["Close"]
    price = float(close.iloc[-1])
    ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
    ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

    # Factor 1 — EMA alignment score (0-3)
    ema_score = 0
    if price > ema20:  ema_score += 1
    if ema20  > ema50: ema_score += 1
    if ema50  > ema200: ema_score += 1

    # Factor 2 — Nifty ADX (trend strength)
    nifty_adx = calc_adx(df)
    trend_strong = nifty_adx > 25

    # Factor 3 — Market breadth
    # Check how many watchlist stocks are above their 200 EMA
    above_200_count = 0
    checked = 0
    sample = WATCHLIST[:30]  # check first 30 stocks for speed
    for ticker in sample:
        try:
            bdf = fetch_data(ticker, period="1y")
            if bdf is not None and len(bdf) >= 200:
                c = bdf["Close"]
                e200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
                if float(c.iloc[-1]) > e200:
                    above_200_count += 1
                checked += 1
        except Exception:
            continue
    breadth_pct = round(above_200_count / checked * 100, 1) if checked > 0 else 50.0

    # Combine all 3 factors into final mood
    bullish_points = 0
    bearish_points = 0

    # EMA alignment
    if ema_score == 3:   bullish_points += 2
    elif ema_score == 2: bullish_points += 1
    elif ema_score == 0: bearish_points += 2
    elif ema_score == 1: bearish_points += 1

    # ADX confirmation
    if trend_strong and ema_score >= 2: bullish_points += 1
    if trend_strong and ema_score <= 1: bearish_points += 1

    # Breadth
    if breadth_pct > 60:   bullish_points += 2
    elif breadth_pct > 50: bullish_points += 1
    elif breadth_pct < 35: bearish_points += 2
    elif breadth_pct < 50: bearish_points += 1

    if bullish_points >= 3 and bullish_points > bearish_points:
        mood = "BULLISH"
    elif bearish_points >= 3 and bearish_points > bullish_points:
        mood = "BEARISH"
    else:
        mood = "NEUTRAL"

    detail = f"EMA score {ema_score}/3 | ADX {round(nifty_adx,1)} | Breadth {breadth_pct}% above 200EMA"
    print(f"   Nifty EMA score : {ema_score}/3")
    print(f"   Nifty ADX       : {round(nifty_adx, 1)} ({'trending' if trend_strong else 'ranging'})")
    print(f"   Market breadth  : {breadth_pct}% stocks above 200 EMA")

    return {"mood": mood, "adx": round(nifty_adx, 1), "breadth": breadth_pct, "detail": detail}

# ──────────────────────────────────────────────
# INDIA VIX
# ──────────────────────────────────────────────
def get_india_vix() -> float:
    try:
        df = yf.download("^INDIAVIX", period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 15.0  # default neutral VIX

# ──────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────
def calc_rsi(series: pd.Series, period=14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    val   = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(val), 2)

def calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_adx(df: pd.DataFrame, period=14) -> float:
    """
    ADX — Average Directional Index.
    Returns 0-100. Above 25 = trending. Below 25 = ranging (avoid trading).
    """
    try:
        high  = df["High"].values
        low   = df["Low"].values
        close = df["Close"].values
        n     = len(close)

        plus_dm  = np.zeros(n)
        minus_dm = np.zeros(n)
        tr_arr   = np.zeros(n)

        for i in range(1, n):
            h_diff = high[i] - high[i-1]
            l_diff = low[i-1] - low[i]
            plus_dm[i]  = h_diff if h_diff > l_diff and h_diff > 0 else 0
            minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0
            tr_arr[i]   = max(high[i] - low[i],
                              abs(high[i] - close[i-1]),
                              abs(low[i]  - close[i-1]))

        def smooth(arr, p):
            result = np.zeros(n)
            result[p] = arr[1:p+1].sum()
            for i in range(p+1, n):
                result[i] = result[i-1] - result[i-1]/p + arr[i]
            return result

        atr_s    = smooth(tr_arr, period)
        plus_s   = smooth(plus_dm, period)
        minus_s  = smooth(minus_dm, period)

        plus_di  = 100 * plus_s  / np.where(atr_s == 0, 1, atr_s)
        minus_di = 100 * minus_s / np.where(atr_s == 0, 1, atr_s)

        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))

        adx = np.zeros(n)
        adx[2*period] = dx[period:2*period+1].mean()
        for i in range(2*period+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

        return round(float(adx[-1]), 2)
    except Exception:
        return 0.0

def calc_supertrend(df: pd.DataFrame, period=10, multiplier=3.0) -> int:
    """Returns +1 (bullish) or -1 (bearish)."""
    try:
        atr   = calc_atr(df, period)
        hl2   = (df["High"] + df["Low"]) / 2
        upper = (hl2 + multiplier * atr).values
        lower = (hl2 - multiplier * atr).values
        close = df["Close"].values

        fu = upper.copy()
        fl = lower.copy()
        direction = np.ones(len(close), dtype=int)

        for i in range(1, len(close)):
            fl[i] = lower[i] if lower[i] > fl[i-1] or close[i-1] < fl[i-1] else fl[i-1]
            fu[i] = upper[i] if upper[i] < fu[i-1] or close[i-1] > fu[i-1] else fu[i-1]
            if direction[i-1] == 1:
                direction[i] = -1 if close[i] < fl[i] else 1
            else:
                direction[i] = 1 if close[i] > fu[i] else -1

        return int(direction[-1])
    except Exception:
        return 0

def calc_macd(series: pd.Series):
    e12  = series.ewm(span=12, adjust=False).mean()
    e26  = series.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return round(float(line.iloc[-1]), 4), round(float(sig.iloc[-1]), 4), round(float(hist.iloc[-1]), 4)

def calc_ema(series: pd.Series, span: int) -> float:
    return round(float(series.ewm(span=span, adjust=False).mean().iloc[-1]), 2)

def calc_vol_spike(vol: pd.Series) -> float:
    avg = vol.iloc[-21:-1].mean()
    return round(float(vol.iloc[-1] / avg), 2) if avg > 0 else 1.0


def calc_gap_risk(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Calculates overnight gap risk based on last N days of gap history.
    Gap = difference between today open and yesterday close.
    Returns avg gap %, max gap %, and risk label.
    """
    opens  = df["Open"].values
    closes = df["Close"].values
    gaps   = []
    for i in range(1, min(lookback + 1, len(opens))):
        gap_pct = abs((opens[i] - closes[i-1]) / closes[i-1] * 100)
        gaps.append(gap_pct)
    if not gaps:
        return {"avg_gap": 0.0, "max_gap": 0.0, "risk": "LOW"}
    avg_gap = round(float(np.mean(gaps)), 2)
    max_gap = round(float(np.max(gaps)), 2)
    if avg_gap > 1.5 or max_gap > 3.0:
        risk = "HIGH"
    elif avg_gap > 0.75 or max_gap > 1.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {"avg_gap": avg_gap, "max_gap": max_gap, "risk": risk}

# ──────────────────────────────────────────────
# ANALYSIS ENGINE v4
# ──────────────────────────────────────────────
def analyse(ticker: str, df: pd.DataFrame, market_mood: str,
            fii_sentiment: str, vix: float) -> dict | None:

    close  = df["Close"]
    volume = df["Volume"]
    symbol = ticker.replace(".NS", "")

    # Liquidity check
    if volume.iloc[-20:].mean() < MIN_AVG_VOLUME:
        return None

    price   = round(float(close.iloc[-1]), 2)
    prev    = round(float(close.iloc[-2]), 2)
    day_chg = round((price - prev) / prev * 100, 2)

    # ── Core indicators ─────────────────────────
    rsi_val   = calc_rsi(close)
    adx_val   = calc_adx(df)
    st_dir    = calc_supertrend(df)       # +1 = bullish, -1 = bearish
    macd_val, sig_val, hist = calc_macd(close)
    ema8      = calc_ema(close, 8)
    ema21     = calc_ema(close, 21)
    ema55     = calc_ema(close, 55)
    ema200    = calc_ema(close, min(200, len(close)-1))
    vspike    = calc_vol_spike(volume)

    gap_info  = calc_gap_risk(df)
    pct_200   = round((price - ema200) / ema200 * 100, 2)
    above_200 = price > ema200
    deep_down = pct_200 < -15

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GATE 1 — ADX filter (most important gate)
    # If market not trending, skip entirely
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if adx_val < ADX_THRESHOLD:
        return None  # ranging market — no trade

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GATE 2 — Volume filter
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if vspike < 0.9:
        return None  # dead volume

    # GATE 3 - Hard RSI gate (mandatory, not optional)
    # RSI must be in valid zone - no exceptions
    is_potential_buy   = above_200 and (st_dir == 1 or ema8 > ema21)
    is_potential_short = not above_200 and (st_dir == -1 or ema8 < ema21)

    if is_potential_buy and not (RSI_BUY_MIN <= rsi_val <= RSI_BUY_MAX):
        return None  # RSI outside 35-55 - rejected

    if is_potential_short and not (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
        return None  # RSI outside 50-72 - rejected

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BUY CONDITIONS — research-backed rules
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    buy_conditions = {
        "above_200ema":    above_200,                          # long-term trend
        "supertrend_bull": st_dir == 1,                        # SuperTrend bullish
        "ema_crossover":   ema8 > ema21 and price > ema55,    # EMA aligned
        "macd_bull":       macd_val > sig_val and hist > 0,   # MACD crossover
        "rsi_pullback":    RSI_BUY_MIN <= rsi_val <= RSI_BUY_MAX,  # RSI in buy zone
        "volume_ok":       vspike >= 1.0,                      # sufficient volume
    }

    # SHORT CONDITIONS
    short_conditions = {
        "below_200ema":    not above_200,                      # long-term downtrend
        "supertrend_bear": st_dir == -1,                       # SuperTrend bearish
        "ema_crossover":   ema8 < ema21 and price < ema55,    # EMA aligned down
        "macd_bear":       macd_val < sig_val and hist < 0,   # MACD crossover down
        "rsi_zone":        RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX,  # RSI in short zone
        "volume_ok":       vspike >= 1.0,
    }

    buy_score  = sum(buy_conditions.values())
    short_score = sum(short_conditions.values())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD RULES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Must have at least 4/6 conditions for BUY
    # Must have at least 4/6 conditions for SHORT
    # Hard block BUY if deep downtrend
    # Hard block SHORT if market is BULLISH and FII buying
    # High VIX = require stronger signals
    min_score = 5 if vix > 18 else 4

    reasons = []

    if buy_score >= min_score and not deep_down:
        # Market-level suppression
        if market_mood == "BEARISH" and fii_sentiment == "FII_BEARISH":
            return None  # don't fight both market and FIIs
        if buy_conditions["above_200ema"]:   reasons.append(f"Above 200 EMA (+{pct_200}%)")
        if buy_conditions["supertrend_bull"]: reasons.append("SuperTrend bullish")
        if buy_conditions["ema_crossover"]:   reasons.append("EMA 8 > 21 > 55 aligned")
        if buy_conditions["macd_bull"]:       reasons.append("MACD bullish crossover")
        if buy_conditions["rsi_pullback"]:    reasons.append(f"RSI pullback in uptrend ({rsi_val})")
        if vspike >= 1.5:                     reasons.append(f"Volume spike {vspike}x")
        signal = "BUY"
        target = round(price * (1 + MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 - STOP_LOSS_PCT / 100), 2)
        score  = buy_score

    elif short_score >= min_score:
        # Market-level suppression
        if market_mood == "BULLISH" and fii_sentiment == "FII_BULLISH":
            return None  # don't short a strong bull market
        if short_conditions["below_200ema"]:   reasons.append(f"Below 200 EMA ({pct_200}%)")
        if short_conditions["supertrend_bear"]: reasons.append("SuperTrend bearish")
        if short_conditions["ema_crossover"]:   reasons.append("EMA 8 < 21 < 55 aligned")
        if short_conditions["macd_bear"]:       reasons.append("MACD bearish crossover")
        if short_conditions["rsi_zone"]:        reasons.append(f"RSI in short zone ({rsi_val})")
        if vspike >= 1.5:                       reasons.append(f"Volume spike {vspike}x")
        signal = "SHORT"
        target = round(price * (1 - MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 + STOP_LOSS_PCT / 100), 2)
        score  = short_score

    else:
        return None

    return {
        "symbol":  symbol,
        "signal":  signal,
        "price":   price,
        "target":  target,
        "sl":      sl,
        "score":   score,
        "adx":     adx_val,
        "rsi":     rsi_val,
        "vol":     vspike,
        "day_chg": day_chg,
        "vs200":   pct_200,
        "gap":     gap_info,
        "reasons": reasons,
    }

# ──────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────
def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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

def build_message(results: list, market_mood: str,
                  mood_data: dict, fii_data: dict, vix: float) -> str:
    now       = datetime.now().strftime("%d %b %Y, %I:%M %p")
    mood_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(market_mood, "🟡")
    fii_icon  = "🟢" if fii_data["fii_net"] > 0 else "🔴"
    vix_icon  = "🔴" if vix > 18 else "🟡" if vix > 14 else "🟢"

    lines = [
        f"📊 <b>NSE Evening Scan v4</b>",
        f"🕐 {esc(now)} IST\n",
        f"<b>Market Context</b>",
        f"  {mood_icon} Nifty Trend: {esc(market_mood)}",
        f"  {fii_icon} FII Flow: ₹{esc(str(fii_data['fii_net']))} Cr  |  DII: ₹{esc(str(fii_data['dii_net']))} Cr",
        f"  {vix_icon} India VIX: {esc(str(vix))}",
        ""
    ]

    buys   = [r for r in results if r["signal"] == "BUY"]
    shorts = [r for r in results if r["signal"] == "SHORT"]

    if buys:
        lines.append("🟢 <b>BUY (LONG)</b>")
        for r in buys:
            reasons_str = esc(" • ".join(r["reasons"]))
            lines.append(
                f"\n<b>{esc(r['symbol'])}</b>  ({r['score']}/6 conditions met)\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (+1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (-0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  ADX: {r['adx']}  |  Vol: {r['vol']}x\n"
                f"  ⚡ Gap Risk: {r['gap']['risk']}  (avg {r['gap']['avg_gap']}%, max {r['gap']['max_gap']}%)\n"
                f"  📝 {reasons_str}"
            )

    if shorts:
        lines.append("\n🔴 <b>SHORT (SELL)</b>")
        for r in shorts:
            reasons_str = esc(" • ".join(r["reasons"]))
            lines.append(
                f"\n<b>{esc(r['symbol'])}</b>  ({r['score']}/6 conditions met)\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (-1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (+0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  ADX: {r['adx']}  |  Vol: {r['vol']}x\n"
                f"  ⚡ Gap Risk: {r['gap']['risk']}  (avg {r['gap']['avg_gap']}%, max {r['gap']['max_gap']}%)\n"
                f"  📝 {reasons_str}\n"
                f"  📦 Instrument: Futures or Put Option"
            )

    if not buys and not shorts:
        lines.append(
            "😐 <b>No signals today.</b>\n\n"
            "ADX filter blocked all ranging stocks. "
            "No clean trending setups found today.\n"
            "Staying out is the right call. 💤"
        )

    lines.append(
        "\n⚠️ <i>Educational only. Not SEBI-registered advice.\n"
        "Always place stop-loss before entering any trade.</i>"
    )
    return "\n".join(lines)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'═'*55}")
    print(f"  NSE Scanner v4 — {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print(f"{'═'*55}\n")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing Telegram credentials")
        sys.exit(1)

    # Market context
    print("🌍 Fetching market context...")
    mood_data   = get_market_mood()
    market_mood = mood_data["mood"]
    fii_data    = get_fii_sentiment()
    vix         = get_india_vix()
    print(f"   Nifty mood : {market_mood}")
    print(f"   FII net    : ₹{fii_data['fii_net']} Cr ({fii_data['sentiment']})")
    print(f"   India VIX  : {vix}")
    print(f"   Min ADX    : {ADX_THRESHOLD} (stocks below this are skipped)\n")

    # Scan
    results = []
    total   = len(WATCHLIST)
    skipped_adx = 0

    for i, ticker in enumerate(WATCHLIST):
        df = fetch_data(ticker)
        if df is not None:
            # Quick ADX check before full analysis
            adx_quick = calc_adx(df)
            if adx_quick < ADX_THRESHOLD:
                skipped_adx += 1
            else:
                rec = analyse(ticker, df, market_mood, fii_data["sentiment"], vix)
                if rec:
                    results.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} scanned — {len(results)} signals, {skipped_adx} skipped (ranging)")

    print(f"\n  Total skipped (ADX < {ADX_THRESHOLD}): {skipped_adx}/{total}")

    # Sort
    buys   = sorted([r for r in results if r["signal"] == "BUY"],  key=lambda x: -x["score"])
    shorts = sorted([r for r in results if r["signal"] == "SHORT"], key=lambda x: -x["score"])
    final  = (buys + shorts)[:MAX_OUTPUT]

    # Console
    print(f"\n{'─'*55}")
    if not final:
        print("😐 No strong signals today.")
    for r in final:
        icon = "🟢" if r["signal"] == "BUY" else "🔴"
        print(f"{icon} {r['symbol']:15} {r['signal']:6}  ₹{r['price']}  →  ₹{r['target']}  SL: ₹{r['sl']}")
        print(f"   ADX: {r['adx']}  RSI: {r['rsi']}  Vol: {r['vol']}x  Score: {r['score']}/6")
        for reason in r["reasons"]:
            print(f"   • {reason}")
    print(f"{'─'*55}\n")

    send_telegram(build_message(final, market_mood, mood_data, fii_data, vix))

if __name__ == "__main__":
    main()
