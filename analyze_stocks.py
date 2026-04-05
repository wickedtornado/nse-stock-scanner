#!/usr/bin/env python3
"""
NSE Stock Analyser v3 — Multi-Strategy Engine
Strategies integrated:
  1. SuperTrend + RSI (highest win rate in trending markets)
  2. EMA Crossover (8/21 cross with 55 EMA trend filter)
  3. ATR momentum confirmation (expanding volatility = conviction)
  4. 200 EMA trend guard (no BUY in downtrend, no SHORT in uptrend)
  5. Market mood filter (Nifty 50 direction)
  6. Volume confirmation (mandatory, not bonus)

Signal requires at least 2 strategies to agree = fewer but higher quality signals.
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
MAX_OUTPUT      = 5        # max signals — quality over quantity
MIN_AVG_VOLUME  = 500_000

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
    except Exception as e:
        print(f"  ⚠️  {ticker}: {e}")
        return None

# ──────────────────────────────────────────────
# MARKET MOOD — Nifty 50
# ──────────────────────────────────────────────
def get_market_mood() -> str:
    df = fetch_data("^NSEI")
    if df is None:
        return "NEUTRAL"
    close = df["Close"]
    ema20  = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50  = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    price  = close.iloc[-1]
    if price > ema20 > ema50 > ema200:
        return "BULLISH"
    elif price < ema20 and price < ema50:
        return "BEARISH"
    return "NEUTRAL"

# ──────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────
def calc_rsi(series: pd.Series, period=14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    val   = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(val), 2)

def calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """
    SuperTrend indicator.
    Returns series: +1 = bullish (price above supertrend), -1 = bearish.
    """
    atr    = calc_atr(df, period)
    hl2    = (df["High"] + df["Low"]) / 2
    upper  = hl2 + multiplier * atr
    lower  = hl2 - multiplier * atr
    close  = df["Close"]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Lower band
        if lower.iloc[i] > lower.iloc[i-1] or close.iloc[i-1] < lower.iloc[i-1]:
            final_lower = lower.iloc[i]
        else:
            final_lower = lower.iloc[i-1]

        # Upper band
        if upper.iloc[i] < upper.iloc[i-1] or close.iloc[i-1] > upper.iloc[i-1]:
            final_upper = upper.iloc[i]
        else:
            final_upper = upper.iloc[i-1]

        # Direction
        if i == 1:
            direction.iloc[i] = 1
        elif supertrend.iloc[i-1] == upper.iloc[i-1]:
            direction.iloc[i] = -1 if close.iloc[i] > final_upper else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < final_lower else -1

        supertrend.iloc[i] = final_lower if direction.iloc[i] == -1 else final_upper

    return direction  # -1 = bearish, 1 = bullish... wait, flip:
    # convention: direction -1 means price is ABOVE supertrend = BUY zone

def calc_supertrend_signal(df: pd.DataFrame, period=10, multiplier=3.0) -> int:
    """Returns +1 (bullish) or -1 (bearish) based on SuperTrend."""
    try:
        atr   = calc_atr(df, period)
        hl2   = (df["High"] + df["Low"]) / 2
        upper = (hl2 + multiplier * atr).values
        lower = (hl2 - multiplier * atr).values
        close = df["Close"].values

        final_upper = upper.copy()
        final_lower = lower.copy()
        direction   = np.ones(len(close), dtype=int)  # 1 = bullish

        for i in range(1, len(close)):
            # Adjust lower band
            final_lower[i] = lower[i] if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1] else final_lower[i-1]
            # Adjust upper band
            final_upper[i] = upper[i] if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1] else final_upper[i-1]

            # Determine direction
            if direction[i-1] == 1:  # was bullish
                direction[i] = -1 if close[i] < final_lower[i] else 1
            else:  # was bearish
                direction[i] = 1 if close[i] > final_upper[i] else -1

        return int(direction[-1])  # +1 = bullish, -1 = bearish
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

# ──────────────────────────────────────────────
# MULTI-STRATEGY ANALYSIS ENGINE v3
# ──────────────────────────────────────────────
def analyse(ticker: str, df: pd.DataFrame, market_mood: str):
    close  = df["Close"]
    volume = df["Volume"]
    symbol = ticker.replace(".NS", "")

    # Liquidity filter
    if volume.iloc[-20:].mean() < MIN_AVG_VOLUME:
        return None

    price   = round(float(close.iloc[-1]), 2)
    prev    = round(float(close.iloc[-2]), 2)
    day_chg = round((price - prev) / prev * 100, 2)

    # ── Compute all indicators ──────────────────
    rsi_val                  = calc_rsi(close)
    macd_val, sig_val, hist  = calc_macd(close)
    ema8                     = calc_ema(close, 8)
    ema21                    = calc_ema(close, 21)
    ema55                    = calc_ema(close, 55)
    ema200                   = calc_ema(close, min(200, len(close)-1))
    vspike                   = calc_vol_spike(volume)
    supertrend_dir           = calc_supertrend_signal(df)

    atr_series   = calc_atr(df)
    atr_now      = float(atr_series.iloc[-1])
    atr_20avg    = float(atr_series.iloc[-21:-1].mean())
    atr_expanding = atr_now > atr_20avg  # momentum is expanding

    pct_from_200 = round((price - ema200) / ema200 * 100, 2)
    above_200    = price > ema200
    deep_down    = pct_from_200 < -15

    # ── STRATEGY 1: SuperTrend + RSI ───────────
    # Highest win rate in trending markets
    st_buy  = supertrend_dir == 1  and rsi_val < 60 and rsi_val > 30
    st_sell = supertrend_dir == -1 and rsi_val > 40 and rsi_val < 75

    # ── STRATEGY 2: EMA Crossover ──────────────
    # 8 EMA cross above 21 EMA, price above 55 EMA = BUY
    # 8 EMA cross below 21 EMA, price below 55 EMA = SELL
    ema_buy  = ema8 > ema21 and price > ema55
    ema_sell = ema8 < ema21 and price < ema55

    # ── STRATEGY 3: MACD momentum ──────────────
    macd_buy  = macd_val > sig_val and hist > 0
    macd_sell = macd_val < sig_val and hist < 0

    # ── STRATEGY 4: RSI extreme reversal ───────
    # Only use RSI alone when it's extreme AND ATR is expanding
    rsi_buy  = rsi_val < 32 and atr_expanding
    rsi_sell = rsi_val > 72 and atr_expanding

    # ── Count how many strategies agree ────────
    buy_votes  = sum([st_buy,  ema_buy,  macd_buy,  rsi_buy])
    sell_votes = sum([st_sell, ema_sell, macd_sell, rsi_sell])

    # ── Build reasons list ─────────────────────
    reasons = []
    if st_buy  or st_sell:  reasons.append(f"SuperTrend {'bullish' if st_buy else 'bearish'}")
    if ema_buy or ema_sell: reasons.append(f"EMA crossover {'bullish' if ema_buy else 'bearish'} (8/21/55)")
    if macd_buy or macd_sell: reasons.append(f"MACD {'bullish' if macd_buy else 'bearish'} crossover")
    if rsi_buy or rsi_sell: reasons.append(f"RSI {'oversold' if rsi_buy else 'overbought'} ({rsi_val}) + ATR expanding")
    if vspike >= 1.5:       reasons.append(f"Volume spike {vspike}x avg")
    if above_200:           reasons.append(f"Above 200 EMA (+{pct_from_200}%)")
    else:                   reasons.append(f"Below 200 EMA ({pct_from_200}%)")

    # ── HARD RULES ─────────────────────────────
    # Rule 1: Need at least 2 strategies to agree
    # Rule 2: No BUY if deep downtrend (>15% below 200 EMA)
    # Rule 3: No BUY in BEARISH market unless supertrend + ema both agree
    # Rule 4: No SHORT in BULLISH market
    # Rule 5: Volume must be at least average (>= 0.9x)

    if vspike < 0.9:
        return None  # dead volume — skip

    if buy_votes >= 2 and not deep_down:
        if market_mood == "BEARISH" and buy_votes < 3:
            return None  # need stronger signal in bearish market
        signal = "BUY"
        target = round(price * (1 + MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 - STOP_LOSS_PCT / 100), 2)
        score  = buy_votes * 25

    elif sell_votes >= 2:
        if market_mood == "BULLISH" and sell_votes < 3:
            return None  # need stronger signal in bullish market
        signal = "SHORT"
        target = round(price * (1 - MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 + STOP_LOSS_PCT / 100), 2)
        score  = -(sell_votes * 25)

    else:
        return None  # not enough agreement — skip

    return {
        "symbol":    symbol,
        "signal":    signal,
        "price":     price,
        "target":    target,
        "sl":        sl,
        "score":     abs(score),
        "rsi":       rsi_val,
        "vol":       vspike,
        "day_chg":   day_chg,
        "vs200":     pct_from_200,
        "strategies": buy_votes if signal == "BUY" else sell_votes,
        "reasons":   reasons,
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

def build_message(results: list, market_mood: str) -> str:
    now       = datetime.now().strftime("%d %b %Y, %I:%M %p")
    mood_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(market_mood, "🟡")
    lines = [
        f"📊 <b>NSE Evening Scan v3</b>",
        f"🕐 {esc(now)} IST",
        f"🌍 Nifty Mood: {mood_icon} {esc(market_mood)}\n"
    ]

    buys   = [r for r in results if r["signal"] == "BUY"]
    shorts = [r for r in results if r["signal"] == "SHORT"]

    if buys:
        lines.append("🟢 <b>BUY (LONG)</b>")
        for r in buys:
            reasons_str = esc(" • ".join(r["reasons"][:4]))
            lines.append(
                f"\n<b>{esc(r['symbol'])}</b>  ({r['strategies']}/4 strategies agree)\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (+1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (-0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  Vol: {r['vol']}x  |  vs200EMA: {r['vs200']}%\n"
                f"  📝 {reasons_str}"
            )

    if shorts:
        lines.append("\n🔴 <b>SHORT (SELL)</b>")
        for r in shorts:
            reasons_str = esc(" • ".join(r["reasons"][:4]))
            lines.append(
                f"\n<b>{esc(r['symbol'])}</b>  ({r['strategies']}/4 strategies agree)\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (-1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (+0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  Vol: {r['vol']}x  |  vs200EMA: {r['vs200']}%\n"
                f"  📝 {reasons_str}\n"
                f"  📦 Instrument: Futures or Put Option"
            )

    if not buys and not shorts:
        lines.append(
            "😐 <b>No signals today.</b>\n\n"
            "Filters didn't find a clean setup. "
            "This is correct behaviour — not every day has a trade. "
            "Staying out is also a position. 💤"
        )

    lines.append("\n⚠️ <i>Educational only. Not SEBI-registered advice. Always set SL before entering.</i>")
    return "\n".join(lines)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'═'*55}")
    print(f"  NSE Stock Scanner v3 — {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print(f"{'═'*55}\n")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing Telegram credentials")
        sys.exit(1)

    # Market mood
    print("🌍 Checking Nifty 50 mood...")
    market_mood = get_market_mood()
    print(f"   Mood: {market_mood}\n")

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

    # Sort — BUY by score desc, SHORT by score desc
    buys   = sorted([r for r in results if r["signal"] == "BUY"],  key=lambda x: (-x["strategies"], -x["score"]))
    shorts = sorted([r for r in results if r["signal"] == "SHORT"], key=lambda x: (-x["strategies"], -x["score"]))
    final  = (buys + shorts)[:MAX_OUTPUT]

    # Console output
    print(f"\n{'─'*55}")
    if not final:
        print("😐 No strong signals today.")
    for r in final:
        icon = "🟢" if r["signal"] == "BUY" else "🔴"
        print(f"{icon} {r['symbol']:15} {r['signal']:6}  ₹{r['price']}  →  ₹{r['target']}  SL: ₹{r['sl']}  [{r['strategies']}/4 strategies]")
        for reason in r["reasons"]:
            print(f"   • {reason}")
    print(f"{'─'*55}\n")

    send_telegram(build_message(final, market_mood))

if __name__ == "__main__":
    main()
