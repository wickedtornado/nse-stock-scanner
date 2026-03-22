#!/usr/bin/env python3
"""
NSE Stock Analyser — Angel One SmartAPI + Telegram Alerts
Runs via Render.com cron job daily at 9:15 AM IST.
Credentials stored as Render Environment Variables.
"""

import os
import sys
import time
import pyotp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from SmartApi import SmartConnect

# ──────────────────────────────────────────────
# CREDENTIALS — from Render Environment Variables
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
ANGEL_API_KEY      = os.environ.get("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD     = os.environ.get("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MIN_PROFIT_PCT = 1.0      # target exactly 1%
STOP_LOSS_PCT  = 0.5      # stop-loss 0.5%
MAX_OUTPUT     = 8        # max recommendations per run
MIN_AVG_VOLUME = 500_000  # filter illiquid stocks

# ──────────────────────────────────────────────
# NSE SYMBOL → Angel One Token MAP
# ──────────────────────────────────────────────
WATCHLIST = {
    "RELIANCE":    "2885",  "TCS":         "11536", "HDFCBANK":    "1333",
    "INFY":        "1594",  "ICICIBANK":   "4963",  "HINDUNILVR":  "1394",
    "SBIN":        "3045",  "BHARTIARTL":  "10604", "ITC":         "1660",
    "KOTAKBANK":   "1922",  "LT":          "11483", "AXISBANK":    "5900",
    "ASIANPAINT":  "236",   "MARUTI":      "10999", "TITAN":       "3506",
    "SUNPHARMA":   "3351",  "WIPRO":       "3787",  "HCLTECH":     "7229",
    "TECHM":       "13538", "BAJFINANCE":  "317",   "BAJAJFINSV":  "16675",
    "TATAMOTORS":  "3456",  "TATASTEEL":   "3499",  "JSWSTEEL":    "11723",
    "HINDALCO":    "1363",  "NTPC":        "11630", "POWERGRID":   "14977",
    "ONGC":        "2475",  "COALINDIA":   "20374", "GRASIM":      "1232",
    "DRREDDY":     "881",   "CIPLA":       "694",   "DIVISLAB":    "10940",
    "EICHERMOT":   "910",   "BPCL":        "526",   "M&M":         "2031",
    "HEROMOTOCO":  "1348",  "INDUSINDBK":  "5258",  "ADANIPORTS":  "15083",
    "APOLLOHOSP":  "157",   "BRITANNIA":   "547",   "TATACONSUM":  "3432",
    "HAVELLS":     "2182",  "PIDILITIND":  "2664",  "DMART":       "13375",
    "SBILIFE":     "21808", "HDFCLIFE":    "467",   "ICICIGI":     "15044",
    "VEDL":        "3063",  "BANKBARODA":  "4668",  "CANBK":       "2763",
    "GAIL":        "1107",  "IOC":         "1624",  "TATAPOWER":   "3426",
    "TORNTPHARM":  "3518",  "LUPIN":       "10440", "BAJAJ-AUTO":  "16669",
    "NAUKRI":      "13751", "PERSISTENT":  "18365", "MPHASIS":     "4503",
    "COFORGE":     "11543", "CHOLAFIN":    "685",   "FEDERALBNK":  "1023",
    "IDFCFIRSTB":  "11957", "VOLTAS":      "3718",  "GODREJCP":    "10099",
    "MARICO":      "4067",  "COLPAL":      "1406",  "DABUR":       "772",
    "BERGEPAINT":  "404",   "ZOMATO":      "21296", "IRCTC":       "13611",
    "ADANIENT":    "25",    "DLF":         "14732", "GODREJPROP":  "10147",
    "OBEROIRLTY":  "20242", "MUTHOOTFIN":  "17971", "SBICARD":     "317",
    "PIIND":       "2662",  "ALKEM":       "13220", "AUROPHARMA":  "275",
}

# ──────────────────────────────────────────────
# ANGEL ONE LOGIN
# ──────────────────────────────────────────────
def angel_login():
    totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
    obj  = SmartConnect(api_key=ANGEL_API_KEY)
    data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
    if not data or data.get("status") is False:
        raise Exception(f"Login failed: {data.get('message', 'Unknown error')}")
    print("✅ Angel One login successful")
    return obj

def fetch_historical(obj, token: str, symbol: str):
    try:
        now       = datetime.now()
        to_date   = now.strftime("%Y-%m-%d %H:%M")
        from_date = (now - timedelta(days=120)).strftime("%Y-%m-%d %H:%M")
        params = {
            "exchange":    "NSE",
            "symboltoken": token,
            "interval":    "ONE_DAY",
            "fromdate":    from_date,
            "todate":      to_date,
        }
        resp = obj.getCandleData(params)
        if not resp or resp.get("status") is False or not resp.get("data"):
            return None
        df = pd.DataFrame(resp["data"], columns=["timestamp","open","high","low","close","volume"])
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df if len(df) >= 30 else None
    except Exception as e:
        print(f"  ⚠️  {symbol}: {e}")
        return None

# ──────────────────────────────────────────────
# TECHNICAL INDICATORS
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
def analyse(symbol: str, df: pd.DataFrame):
    close  = df["close"]
    volume = df["volume"]

    if volume.iloc[-20:].mean() < MIN_AVG_VOLUME:
        return None

    price    = round(float(close.iloc[-1]), 2)
    prev     = round(float(close.iloc[-2]), 2)
    day_chg  = round((price - prev) / prev * 100, 2)

    rsi_val                  = calc_rsi(close)
    macd_val, sig_val, hist  = calc_macd(close)
    bb_upper, bb_lower, pctb = calc_bollinger(close)
    ema9                     = calc_ema(close, 9)
    ema21                    = calc_ema(close, 21)
    vspike                   = calc_vol_spike(volume)

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

    # EMA trend
    if price > ema9 > ema21:
        score += 15; reasons.append("Uptrend: price > EMA9 > EMA21")
    elif price < ema9 < ema21:
        score -= 10; reasons.append("Downtrend: price < EMA9 < EMA21")

    # Volume
    if vspike > 1.5:
        score += 10; reasons.append(f"Volume spike {vspike}x")

    if score >= 45:
        signal = "BUY"
        target = round(price * (1 + MIN_PROFIT_PCT / 100), 2)
        sl     = round(price * (1 - STOP_LOSS_PCT / 100), 2)
    elif score <= -20:
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

def build_message(results: list) -> str:
    now  = datetime.now().strftime("%d %b %Y, %I:%M %p")
    hour = datetime.now().hour
    scan = "🌅 Morning" if hour < 12 else "🌆 End-of-Day"
    lines = [f"📊 <b>{scan} Stock Scan</b>", f"🕐 {now} IST\n"]

    buys   = [r for r in results if r["signal"] == "BUY"]
    shorts = [r for r in results if r["signal"] == "SHORT"]

    if buys:
        lines.append("🟢 <b>BUY (LONG)</b>")
        for r in buys:
            lines.append(
                f"\n<b>{r['symbol']}</b>  (Score: {r['score']})\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (+1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (-0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  Vol: {r['vol']}x  |  Day: {r['day_chg']}%\n"
                f"  📝 {' • '.join(r['reasons'][:2])}"
            )

    if shorts:
        lines.append("\n🔴 <b>SHORT (SELL)</b>")
        for r in shorts:
            lines.append(
                f"\n<b>{r['symbol']}</b>  (Score: {r['score']})\n"
                f"  💰 Entry: ₹{r['price']}\n"
                f"  🎯 Target: ₹{r['target']}  (-1%)\n"
                f"  🛑 Stop Loss: ₹{r['sl']}  (+0.5%)\n"
                f"  📊 RSI: {r['rsi']}  |  Vol: {r['vol']}x  |  Day: {r['day_chg']}%\n"
                f"  📝 {' • '.join(r['reasons'][:2])}\n"
                f"  📦 Use: Futures or Put Option"
            )

    lines.append("\n⚠️ <i>Educational only. Not SEBI-registered advice. Always set SL.</i>")
    return "\n".join(lines)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'═'*55}")
    print(f"  NSE Stock Scanner — {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print(f"{'═'*55}\n")

    # Validate credentials
    missing = [k for k, v in {
        "ANGEL_API_KEY": ANGEL_API_KEY,
        "ANGEL_CLIENT_ID": ANGEL_CLIENT_ID,
        "ANGEL_PASSWORD": ANGEL_PASSWORD,
        "ANGEL_TOTP_SECRET": ANGEL_TOTP_SECRET,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }.items() if not v]

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Login
    try:
        obj = angel_login()
    except Exception as e:
        msg = f"❌ Angel One login failed: {e}"
        print(msg)
        send_telegram(f"🚨 <b>Stock Scanner Error</b>\n{msg}")
        sys.exit(1)

    # Scan
    results = []
    total   = len(WATCHLIST)
    for i, (symbol, token) in enumerate(WATCHLIST.items()):
        df = fetch_historical(obj, token, symbol)
        if df is not None:
            rec = analyse(symbol, df)
            if rec:
                results.append(rec)
        time.sleep(0.4)  # avoid Angel One rate limiting
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} scanned — {len(results)} signals so far")

    # Sort and trim
    buys   = sorted([r for r in results if r["signal"] == "BUY"],   key=lambda x: -x["score"])
    shorts = sorted([r for r in results if r["signal"] == "SHORT"],  key=lambda x: x["score"])
    final  = (buys + shorts)[:MAX_OUTPUT]

    if not final:
        print("\n😐 No strong signals today.")
        send_telegram("📊 <b>Stock Scan Complete</b>\n\nNo strong signals today. Market is ranging. Stay patient. 💤")
        return

    # Print to console (visible in Render logs)
    print(f"\n{'─'*55}")
    for r in final:
        icon = "🟢" if r["signal"] == "BUY" else "🔴"
        print(f"{icon} {r['symbol']:15} {r['signal']:6}  ₹{r['price']}  →  ₹{r['target']}  SL: ₹{r['sl']}")
        for reason in r["reasons"]:
            print(f"   • {reason}")
    print(f"{'─'*55}\n")

    send_telegram(build_message(final))

if __name__ == "__main__":
    main()
