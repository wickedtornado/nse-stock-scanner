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
# SECTOR MAP — stock to sector mapping
# ──────────────────────────────────────────────
SECTOR_MAP = {
    # IT
    "TCS.NS":        "IT", "INFY.NS":       "IT", "WIPRO.NS":      "IT",
    "HCLTECH.NS":    "IT", "TECHM.NS":      "IT", "PERSISTENT.NS": "IT",
    "MPHASIS.NS":    "IT", "COFORGE.NS":    "IT", "NAUKRI.NS":     "IT",

    # BANKING
    "HDFCBANK.NS":   "BANK", "ICICIBANK.NS":  "BANK", "KOTAKBANK.NS":  "BANK",
    "SBIN.NS":       "BANK", "AXISBANK.NS":   "BANK", "INDUSINDBK.NS": "BANK",
    "BANKBARODA.NS": "BANK", "CANBK.NS":      "BANK", "FEDERALBNK.NS": "BANK",
    "IDFCFIRSTB.NS": "BANK",

    # FINANCIAL SERVICES
    "BAJFINANCE.NS": "FINSERV", "BAJAJFINSV.NS": "FINSERV", "CHOLAFIN.NS":   "FINSERV",
    "MUTHOOTFIN.NS": "FINSERV", "SBILIFE.NS":    "FINSERV", "HDFCLIFE.NS":   "FINSERV",
    "ICICIGI.NS":    "FINSERV",

    # PHARMA
    "SUNPHARMA.NS":  "PHARMA", "DRREDDY.NS":    "PHARMA", "CIPLA.NS":      "PHARMA",
    "DIVISLAB.NS":   "PHARMA", "TORNTPHARM.NS": "PHARMA", "LUPIN.NS":      "PHARMA",
    "ALKEM.NS":      "PHARMA", "AUROPHARMA.NS": "PHARMA",

    # AUTO
    "MARUTI.NS":     "AUTO", "TATAMOTORS.NS": "AUTO", "M&M.NS":        "AUTO",
    "HEROMOTOCO.NS": "AUTO", "EICHERMOT.NS":  "AUTO", "BAJAJ-AUTO.NS": "AUTO",

    # METALS
    "TATASTEEL.NS":  "METALS", "JSWSTEEL.NS":   "METALS", "HINDALCO.NS":   "METALS",
    "VEDL.NS":       "METALS", "COALINDIA.NS":  "METALS",

    # OIL & GAS
    "RELIANCE.NS":   "OILGAS", "ONGC.NS":       "OILGAS", "BPCL.NS":       "OILGAS",
    "IOC.NS":        "OILGAS", "GAIL.NS":        "OILGAS",

    # FMCG
    "HINDUNILVR.NS": "FMCG", "ITC.NS":        "FMCG", "BRITANNIA.NS":  "FMCG",
    "TATACONSUM.NS": "FMCG", "MARICO.NS":     "FMCG", "COLPAL.NS":     "FMCG",
    "DABUR.NS":      "FMCG", "GODREJCP.NS":   "FMCG",

    # INFRA / CAPITAL GOODS
    "LT.NS":         "INFRA", "ADANIPORTS.NS": "INFRA", "NTPC.NS":       "INFRA",
    "POWERGRID.NS":  "INFRA", "TATAPOWER.NS":  "INFRA", "GRASIM.NS":     "INFRA",

    # CONSUMER / RETAIL
    "TITAN.NS":      "CONSUMER", "ASIANPAINT.NS": "CONSUMER", "HAVELLS.NS":    "CONSUMER",
    "PIDILITIND.NS": "CONSUMER", "VOLTAS.NS":     "CONSUMER", "BERGEPAINT.NS": "CONSUMER",
    "DMART.NS":      "CONSUMER",

    # REALTY
    "DLF.NS":        "REALTY", "GODREJPROP.NS": "REALTY", "OBEROIRLTY.NS": "REALTY",

    # TELECOM
    "BHARTIARTL.NS": "TELECOM",

    # HEALTHCARE
    "APOLLOHOSP.NS": "HEALTHCARE",

    # CONSUMER INTERNET
    "ZOMATO.NS":     "INTERNET", "IRCTC.NS":      "INTERNET",

    # OTHERS
    "PIIND.NS":      "CHEMICALS", "ADANIENT.NS":   "CONGLOMERATE",
}

# Sector ETF/Index tickers for momentum check
SECTOR_ETFS = {
    "IT":       "^CNXIT",       # Nifty IT
    "BANK":     "^NSEBANK",     # Bank Nifty
    "PHARMA":   "^CNXPHARMA",   # Nifty Pharma
    "AUTO":     "^CNXAUTO",     # Nifty Auto
    "METALS":   "^CNXMETAL",    # Nifty Metal
    "FMCG":     "^CNXFMCG",     # Nifty FMCG
    "FINSERV":  "^CNXFINANCE",  # Nifty Financial Services
    "INFRA":    "^CNXINFRA",    # Nifty Infra
    "OILGAS":   "^CNXENERGY",   # Nifty Energy
    "REALTY":   "^CNXREALTY",   # Nifty Realty
}

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


def get_sector_momentum(sector: str) -> str:
    """
    Checks sector index momentum using EMA alignment.
    Returns BULLISH, BEARISH, or NEUTRAL.
    Cached per run — fetched once per sector, not per stock.
    """
    etf = SECTOR_ETFS.get(sector)
    if not etf:
        return "NEUTRAL"  # unknown sector — don't block
    try:
        df = yf.download(etf, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return "NEUTRAL"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].astype(float)
        price = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        if price > ema20 > ema50:
            return "BULLISH"
        elif price < ema20 and price < ema50:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

# Cache sector momentum so we don't re-fetch for every stock in same sector
_sector_cache: dict = {}

def get_sector_momentum_cached(sector: str) -> str:
    if sector not in _sector_cache:
        _sector_cache[sector] = get_sector_momentum(sector)
    return _sector_cache[sector]

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



def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """
    Detects 7 key candlestick patterns on the last 3 candles.
    Returns dict with detected patterns and overall bias (BULLISH/BEARISH/NEUTRAL).
    
    Patterns:
      Bullish: Hammer, Bullish Engulfing, Morning Star
      Bearish: Shooting Star, Bearish Engulfing, Evening Star
      Neutral: Doji (used as filter only)
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    if len(df) < 3:
        return {"patterns": [], "bias": "NEUTRAL"}

    # Last 3 candles
    o1, h1, l1, c1 = o[-3], h[-3], l[-3], c[-3]  # 3 days ago
    o2, h2, l2, c2 = o[-2], h[-2], l[-2], c[-2]  # yesterday
    o3, h3, l3, c3 = o[-1], h[-1], l[-1], c[-1]  # today (latest)

    patterns = []

    # ── Helpers ─────────────────────────────────
    def body(op, cl):     return abs(cl - op)
    def upper_wick(op, cl, hi): return hi - max(op, cl)
    def lower_wick(op, cl, lo): return min(op, cl) - lo
    def is_bullish(op, cl): return cl > op
    def is_bearish(op, cl): return cl < op

    body3 = body(o3, c3)
    body2 = body(o2, c2)
    body1 = body(o1, c1)
    uw3   = upper_wick(o3, c3, h3)
    lw3   = lower_wick(o3, c3, l3)
    uw2   = upper_wick(o2, c2, h2)
    lw2   = lower_wick(o2, c2, l2)
    rng3  = h3 - l3  # full candle range

    # ── HAMMER (Bullish reversal) ────────────────
    # Small body near top, long lower wick >= 2x body, little upper wick
    if (body3 > 0 and
        lw3 >= 2 * body3 and
        uw3 <= 0.3 * body3 and
        rng3 > 0):
        patterns.append("Hammer")

    # ── SHOOTING STAR (Bearish reversal) ─────────
    # Small body near bottom, long upper wick >= 2x body, little lower wick
    if (body3 > 0 and
        uw3 >= 2 * body3 and
        lw3 <= 0.3 * body3 and
        rng3 > 0):
        patterns.append("Shooting Star")

    # ── DOJI (Indecision) ────────────────────────
    # Very small body relative to range
    if rng3 > 0 and body3 <= 0.1 * rng3:
        patterns.append("Doji")

    # ── BULLISH ENGULFING ────────────────────────
    # Today bullish candle completely engulfs yesterday bearish candle
    if (is_bearish(o2, c2) and
        is_bullish(o3, c3) and
        o3 <= c2 and
        c3 >= o2 and
        body3 > body2):
        patterns.append("Bullish Engulfing")

    # ── BEARISH ENGULFING ────────────────────────
    # Today bearish candle completely engulfs yesterday bullish candle
    if (is_bullish(o2, c2) and
        is_bearish(o3, c3) and
        o3 >= c2 and
        c3 <= o2 and
        body3 > body2):
        patterns.append("Bearish Engulfing")

    # ── MORNING STAR (Strong bullish reversal) ───
    # Day1: big bearish, Day2: small body (star), Day3: big bullish
    if (is_bearish(o1, c1) and
        body1 > 0 and
        body2 <= 0.3 * body1 and       # small star
        is_bullish(o3, c3) and
        c3 >= (o1 + c1) / 2):           # closes above midpoint of day1
        patterns.append("Morning Star")

    # ── EVENING STAR (Strong bearish reversal) ───
    # Day1: big bullish, Day2: small body (star), Day3: big bearish
    if (is_bullish(o1, c1) and
        body1 > 0 and
        body2 <= 0.3 * body1 and       # small star
        is_bearish(o3, c3) and
        c3 <= (o1 + c1) / 2):           # closes below midpoint of day1
        patterns.append("Evening Star")

    # ── Determine overall bias ───────────────────
    bullish_patterns = {"Hammer", "Bullish Engulfing", "Morning Star"}
    bearish_patterns = {"Shooting Star", "Bearish Engulfing", "Evening Star"}

    bull_count = sum(1 for p in patterns if p in bullish_patterns)
    bear_count = sum(1 for p in patterns if p in bearish_patterns)

    if bull_count > bear_count:
        bias = "BULLISH"
    elif bear_count > bull_count:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {"patterns": patterns, "bias": bias}



def check_earnings_soon(ticker: str, days_ahead: int = 3) -> bool:
    """
    Returns True if earnings are expected within next N days.
    Uses yfinance calendar data.
    Avoids trading 1-2 days before results — gap risk is extreme.
    """
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar
        if cal is None:
            return False
        # calendar can be a dict or DataFrame depending on yfinance version
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if earnings_date is None:
                return False
            if isinstance(earnings_date, list):
                earnings_date = earnings_date[0]
        elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
            earnings_date = cal["Earnings Date"].iloc[0]
        else:
            return False

        if earnings_date is None:
            return False

        # Convert to date
        from datetime import date
        if hasattr(earnings_date, "date"):
            ed = earnings_date.date()
        else:
            ed = pd.Timestamp(earnings_date).date()

        today = date.today()
        delta = (ed - today).days
        return 0 <= delta <= days_ahead

    except Exception:
        return False

def calc_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Finds key support and resistance levels using recent swing highs/lows.
    A swing high = candle whose high is higher than 2 candles on each side.
    A swing low  = candle whose low  is lower  than 2 candles on each side.
    Returns nearest resistance above price and nearest support below price.
    """
    highs  = df["High"].values[-lookback:]
    lows   = df["Low"].values[-lookback:]
    closes = df["Close"].values[-lookback:]
    price  = float(closes[-1])

    swing_highs = []
    swing_lows  = []

    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and            highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and            lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])

    # Nearest resistance above current price
    resistances = [h for h in swing_highs if h > price * 1.001]
    nearest_resistance = round(float(min(resistances)), 2) if resistances else None

    # Nearest support below current price
    supports = [l for l in swing_lows if l < price * 0.999]
    nearest_support = round(float(max(supports)), 2) if supports else None

    # Distance to resistance as % from current price
    dist_to_resistance = round((nearest_resistance - price) / price * 100, 2) if nearest_resistance else None

    return {
        "resistance":      nearest_resistance,
        "support":         nearest_support,
        "dist_resistance": dist_to_resistance,  # % away from price
    }

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
# CANDLESTICK PATTERN DETECTION
# ──────────────────────────────────────────────
def detect_candle_patterns(df: pd.DataFrame) -> dict:
    """
    Detects 8 high-reliability candlestick patterns on daily data.
    Bullish: Hammer, Bullish Engulfing, Morning Star, Three White Soldiers
    Bearish: Shooting Star, Bearish Engulfing, Evening Star, Three Black Crows
    Returns dict with detected patterns and direction (+1 bullish, -1 bearish, 0 none)
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    if len(c) < 3:
        return {"patterns": [], "bias": 0}

    patterns = []
    i = len(c) - 1  # latest candle index

    body       = lambda idx: abs(c[idx] - o[idx])
    upper_wick = lambda idx: h[idx] - max(c[idx], o[idx])
    lower_wick = lambda idx: min(c[idx], o[idx]) - l[idx]
    is_green   = lambda idx: c[idx] > o[idx]
    is_red     = lambda idx: c[idx] < o[idx]
    candle_range = lambda idx: h[idx] - l[idx]

    # ── BULLISH PATTERNS ───────────────────────

    # 1. Hammer — small body at top, long lower wick >= 2x body, little upper wick
    if (body(i) > 0 and
        lower_wick(i) >= 2 * body(i) and
        upper_wick(i) <= 0.3 * body(i) and
        candle_range(i) > 0):
        patterns.append("Hammer 🔨")

    # 2. Bullish Engulfing — red candle followed by green that fully engulfs it
    if (i >= 1 and
        is_red(i-1) and is_green(i) and
        o[i] <= c[i-1] and
        c[i] >= o[i-1] and
        body(i) > body(i-1)):
        patterns.append("Bullish Engulfing 📈")

    # 3. Morning Star — red candle, small body (doji-like), then strong green
    if (i >= 2 and
        is_red(i-2) and
        body(i-1) <= 0.3 * body(i-2) and  # small middle candle
        is_green(i) and
        c[i] > (o[i-2] + c[i-2]) / 2):    # green closes above midpoint of first red
        patterns.append("Morning Star ⭐")

    # 4. Three White Soldiers — 3 consecutive green candles, each closing higher
    if (i >= 2 and
        is_green(i) and is_green(i-1) and is_green(i-2) and
        c[i] > c[i-1] > c[i-2] and
        o[i] > o[i-1] > o[i-2] and
        body(i) > 0 and body(i-1) > 0 and body(i-2) > 0):
        patterns.append("Three White Soldiers 💪")

    # ── BEARISH PATTERNS ───────────────────────

    # 5. Shooting Star — small body at bottom, long upper wick >= 2x body
    if (body(i) > 0 and
        upper_wick(i) >= 2 * body(i) and
        lower_wick(i) <= 0.3 * body(i) and
        candle_range(i) > 0):
        patterns.append("Shooting Star 🌠")

    # 6. Bearish Engulfing — green candle followed by red that fully engulfs it
    if (i >= 1 and
        is_green(i-1) and is_red(i) and
        o[i] >= c[i-1] and
        c[i] <= o[i-1] and
        body(i) > body(i-1)):
        patterns.append("Bearish Engulfing 📉")

    # 7. Evening Star — green candle, small body, then strong red
    if (i >= 2 and
        is_green(i-2) and
        body(i-1) <= 0.3 * body(i-2) and
        is_red(i) and
        c[i] < (o[i-2] + c[i-2]) / 2):
        patterns.append("Evening Star 🌆")

    # 8. Three Black Crows — 3 consecutive red candles, each closing lower
    if (i >= 2 and
        is_red(i) and is_red(i-1) and is_red(i-2) and
        c[i] < c[i-1] < c[i-2] and
        o[i] < o[i-1] < o[i-2] and
        body(i) > 0 and body(i-1) > 0 and body(i-2) > 0):
        patterns.append("Three Black Crows 🦅")

    # ── Determine overall bias ─────────────────
    bullish_patterns = ["Hammer 🔨", "Bullish Engulfing 📈", "Morning Star ⭐", "Three White Soldiers 💪"]
    bearish_patterns = ["Shooting Star 🌠", "Bearish Engulfing 📉", "Evening Star 🌆", "Three Black Crows 🦅"]

    bull_count = sum(1 for p in patterns if p in bullish_patterns)
    bear_count = sum(1 for p in patterns if p in bearish_patterns)

    if bull_count > bear_count:
        bias = 1
    elif bear_count > bull_count:
        bias = -1
    else:
        bias = 0

    return {"patterns": patterns, "bias": bias, "bull": bull_count, "bear": bear_count}

# ──────────────────────────────────────────────
# ANALYSIS ENGINE v4
# ──────────────────────────────────────────────
def analyse(ticker: str, df: pd.DataFrame, market_mood: str,
            fii_sentiment: str, vix: float, adx_val: float = 0.0) -> dict | None:

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
    # adx_val passed in from main() — no duplicate calculation
    st_dir    = calc_supertrend(df)       # +1 = bullish, -1 = bearish
    macd_val, sig_val, hist = calc_macd(close)
    ema8      = calc_ema(close, 8)
    ema21     = calc_ema(close, 21)
    ema55     = calc_ema(close, 55)
    ema200    = calc_ema(close, min(200, len(close)-1))
    vspike    = calc_vol_spike(volume)

    gap_info      = calc_gap_risk(df)
    sr_info       = calc_support_resistance(df)
    earnings_soon = check_earnings_soon(ticker)
    sector        = SECTOR_MAP.get(ticker, "OTHER")
    sector_mood   = get_sector_momentum_cached(sector)
    candle    = detect_candlestick_patterns(df)
    candles   = detect_candle_patterns(df)
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

    # GATE 3 - Sector momentum gate
    # Don't BUY a stock in a bearish sector — swimming against the tide
    # Don't SHORT a stock in a bullish sector
    # Exception: if sector is NEUTRAL, allow both
    is_likely_buy   = above_200 and st_dir == 1
    is_likely_short = not above_200 and st_dir == -1

    if is_likely_buy and sector_mood == "BEARISH":
        return None  # whole sector is weak — individual BUY unlikely to work

    if is_likely_short and sector_mood == "BULLISH":
        return None  # whole sector is strong — individual SHORT unlikely to work

    # GATE 4 - Earnings gate
    # Never trade 1-3 days before earnings — extreme gap risk
    if earnings_soon:
        return None  # earnings coming up — skip regardless of signal

    # GATE 5 - Candlestick pattern gate
    # If a strong bearish pattern forms today, block BUY
    # If a strong bullish pattern forms today, block SHORT
    # Doji = indecision = reduce confidence but don't block
    strong_bearish_candles = {"Bearish Engulfing", "Evening Star", "Shooting Star"}
    strong_bullish_candles = {"Bullish Engulfing", "Morning Star", "Hammer"}
    candle_patterns = set(candle["patterns"])

    if any(p in strong_bearish_candles for p in candle_patterns) and above_200:
        return None  # bearish candle on a BUY candidate — contradicts signal

    if any(p in strong_bullish_candles for p in candle_patterns) and not above_200:
        return None  # bullish candle on a SHORT candidate — contradicts signal

    # GATE 6 - Hard RSI gate (mandatory, not optional)
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
        # Candlestick confirmation
        if candles["bias"] == -1:
            return None  # bearish candle pattern on a BUY setup — skip
        if candles["patterns"]:
            reasons.append(f"Candle: {', '.join(candles['patterns'])}")
        if candle["bias"] == "BULLISH":       reasons.append(f"Candle: {', '.join(candle['patterns'])}")
        if buy_conditions["above_200ema"]:   reasons.append(f"Above 200 EMA (+{pct_200}%)")
        if buy_conditions["supertrend_bull"]: reasons.append("SuperTrend bullish")
        if buy_conditions["ema_crossover"]:   reasons.append("EMA 8 > 21 > 55 aligned")
        if buy_conditions["macd_bull"]:       reasons.append("MACD bullish crossover")
        if buy_conditions["rsi_pullback"]:    reasons.append(f"RSI pullback in uptrend ({rsi_val})")
        if vspike >= 1.5:                     reasons.append(f"Volume spike {vspike}x")
        # Dynamic target based on signal strength
        if buy_score == 6:   profit_pct = 2.0
        elif buy_score == 5: profit_pct = 1.5
        else:                profit_pct = 1.0

        # S/R check — block BUY if resistance wall is closer than target
        if sr_info["dist_resistance"] is not None:
            if sr_info["dist_resistance"] < profit_pct:
                return None  # resistance wall before target — skip

        signal = "BUY"
        target = round(price * (1 + profit_pct / 100), 2)
        sl     = round(price * (1 - STOP_LOSS_PCT / 100), 2)
        score  = buy_score

    elif short_score >= min_score:
        # Market-level suppression
        if market_mood == "BULLISH" and fii_sentiment == "FII_BULLISH":
            return None  # don't short a strong bull market
        # Candlestick confirmation
        if candles["bias"] == 1:
            return None  # bullish candle pattern on a SHORT setup — skip
        if candles["patterns"]:
            reasons.append(f"Candle: {', '.join(candles['patterns'])}")
        if candle["bias"] == "BEARISH":         reasons.append(f"Candle: {', '.join(candle['patterns'])}")
        if short_conditions["below_200ema"]:   reasons.append(f"Below 200 EMA ({pct_200}%)")
        if short_conditions["supertrend_bear"]: reasons.append("SuperTrend bearish")
        if short_conditions["ema_crossover"]:   reasons.append("EMA 8 < 21 < 55 aligned")
        if short_conditions["macd_bear"]:       reasons.append("MACD bearish crossover")
        if short_conditions["rsi_zone"]:        reasons.append(f"RSI in short zone ({rsi_val})")
        if vspike >= 1.5:                       reasons.append(f"Volume spike {vspike}x")
        signal = "SHORT"
        # Dynamic target based on signal strength
        if short_score == 6:   profit_pct = 2.0
        elif short_score == 5: profit_pct = 1.5
        else:                  profit_pct = 1.0
        target = round(price * (1 - profit_pct / 100), 2)
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
        "candles": candles["patterns"],
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
                f"  🎯 Target: ₹{r['target']}  (+{r.get('profit_pct', 1.0)}%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (-0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  ADX: {r['adx']}  |  Vol: {r['vol']}x\n"
                f"  🏭 Sector: {esc(r['sector'])} ({esc(r['sector_mood'])})\n"
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
                f"  🎯 Target: ₹{r['target']}  (-{r.get('profit_pct', 1.0)}%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (+0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  ADX: {r['adx']}  |  Vol: {r['vol']}x\n"
                f"  🏭 Sector: {esc(r['sector'])} ({esc(r['sector_mood'])})\n"
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
                rec = analyse(ticker, df, market_mood, fii_data["sentiment"], vix, adx_val=adx_quick)
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
